import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from os.path import join
from eoas_pyutils.io_utils.io_common import create_folder

def train_model(model, optimizer, loss_func, train_loader, val_loader, max_num_epochs, 
                 model_name='', device='gpu', patience=10, output_folder='training'):
    '''
    Main function in charge of training a model
    :param model:
    :param optimizer:
    :param loss_func:
    :param train_loader:
    :param val_loader:
    :param max_num_epochs:
    :param device:
    :return:
    '''
    print("Training model...")
    cur_time = datetime.now()
    model_name = f'{model_name}_{cur_time.strftime("%Y-%m-%d_%H:%M")}'

    output_folder = join(output_folder, model_name)
    create_folder(output_folder)
    create_folder(join(output_folder, 'models'))
    create_folder(join(output_folder, 'logs'))

    writer = SummaryWriter(join(model_name,output_folder, 'logs'))
    min_val_loss = 1e10

    # Track the number of epochs since the last improvement
    epochs_no_improve = 0
    for epoch in range(max_num_epochs):
        model.train()

        # Loop over each batch from the training set (to update the model)
        sum_training_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f'{batch_idx}/{len(train_loader.dataset)}', end='\r')
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            sum_training_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        cur_val_loss = 0

        # Loop over each batch from the validation set (to evaluate the model)
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                cur_val_loss += loss_func(output, target).item()

        cur_val_loss /= len(val_loader.dataset)
        if cur_val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = cur_val_loss
            torch.save(model.state_dict(), join(output_folder, 'models', f'{model_name}_epoch:{epoch:03d}_{min_val_loss:0.8f}.pt'))
        else:
            epochs_no_improve += 1

        # ==================  Saving data for tensorboard ============
        # Normal loss
        mean_train_loss = sum_training_loss/len(train_loader.dataset)
        writer.add_scalar('Loss/train', mean_train_loss, epoch)
        writer.add_scalar('Loss/val', cur_val_loss, epoch)
        writer.add_scalars('T&V', {'training':mean_train_loss,
                                          'validation':cur_val_loss}, global_step=epoch)

        # Saving the graph of the model (only once)
        if epoch == 0:
            writer.add_graph(model, data)

        # Stopping the model if there is no improvement
        if epochs_no_improve >= patience:
            print('Early stopping!')
            break

        print(f'Epoch: {epoch+1}, Val loss: {cur_val_loss:.6f} Training loss: {mean_train_loss :.4f}')

    print("Done!")
    writer.close()
    return model