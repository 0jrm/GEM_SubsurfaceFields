import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as ccm

def plot_ts_profiles(temp_profiles, sal_profiles, start_profile, depths=range(0, -2001, -1)):
    """
    Plots the temperature and salinity profiles for each of the four profiles in the dataset.

    Parameters
    ----------
    temp_profiles : numpy.ndarray
    sal_profiles : numpy.ndarray
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot each profile
    for i, ax in enumerate(axs.flat):
        # Create a second x-axis for salinity
        ax2 = ax.twiny()
        # Plot temperature profile
        ax.plot(temp_profiles[:, start_profile], depths, 'r-', label='Temperature')
        # Plot salinity profile
        ax2.plot(sal_profiles[:, start_profile], depths, 'g-', label='Salinity')
        # Add labels and title
        ax.set_title(f'Profile {start_profile}')
        ax.set_xlabel('Temperature (°C)', color='r')
        ax.set_ylabel('Depth (m)')
        ax2.set_xlabel('Salinity (PSU)', color='g')
        # Invert y-axis so depth increases downwards
        ax.set_ylim(reversed(ax.get_ylim()))
        ax2.set_ylim(reversed(ax2.get_ylim()))
        # Add legend
        ax.legend(loc='center right')
        ax2.legend(loc='lower right')
        # Make the font of the x axis blue
        ax.tick_params(axis='x', colors='r')
        ax2.tick_params(axis='x', colors='g')

        start_profile += 1

    # Make the layout more tight
    plt.tight_layout()
    plt.show()

def plot_single_ts_profile(t, s,  depths=range(0, -2001, -1), title='Profile',
                           labelone='Temperature', labeltwo='Salinity'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    # Create a second x-axis for salinity
    ax2 = ax.twiny()
    # Plot temperature profile
    ax.plot(t, depths, 'r-', label='Temperature')
    # Plot salinity profile
    ax2.plot(s, depths, 'g-', label='Salinity')
    # Add labels and title
    ax.set_title(f'{title}')
    ax.set_xlabel('Temperature (°C)', color='r')
    ax.set_ylabel('Depth (m)')
    ax2.set_xlabel('Salinity (PSU)', color='g')
    # Invert y-axis so depth increases downwards
    ax.set_ylim(reversed(ax.get_ylim()))
    ax2.set_ylim(reversed(ax2.get_ylim()))
    # Add legend
    ax.legend(loc='center right')
    ax2.legend(loc='lower right')
    # Make the font of the x axis blue
    ax.tick_params(axis='x', colors='r')
    ax2.tick_params(axis='x', colors='g')

    # Make the layout more tight
    plt.tight_layout()
    plt.show()

def compare_profiles(p1, p2,  depths=range(0, -2001, -1), title='Profile',
                           labelone='Original', labeltwo='Preprocessed', figsize=10, same_parameter=False):

    fig, ax = plt.subplots(1, 1, figsize=(figsize, int(figsize*1.2)))

    # Create a second x-axis for salinity
    # ax2 = ax.twiny()
    # Plot temperature profile
    ax.plot(p1, depths, 'r-', label=labelone)
    # Plot salinity profile
    ax.plot(p2, depths, 'g-', label=labeltwo)
    # Add labels and title
    ax.set_title(f'{title}')
    ax.set_xlabel(labelone, color='r')
    ax.set_ylabel('Depth (m)')
    # ax2.set_xlabel(labeltwo, color='g')
    # Invert y-axis so depth increases downwards
    # ax.set_ylim(reversed(ax.get_ylim()))
    # ax2.set_ylim(reversed(ax2.get_ylim()))
    # Add legend
    ax.legend(loc='center right')
    # ax2.legend(loc='lower right')
    # Make the font of the x axis blue
    ax.tick_params(axis='x', colors='r')
    # ax2.tick_params(axis='x', colors='g')
    
    # if same_parameter:
    #     # Find the global min and max values
    #     min_p1 = np.min(p1)
    #     min_p2 = np.min(p2)
    #     max_p1 = np.max(p1)
    #     max_p2 = np.max(p2)
    #     global_min = np.floor(np.min(min_p1, min_p2))
    #     global_max = np.ceil(np.max(max_p1, max_p2))
    #     # Set x-axis limits for both ax and ax2
    #     ax.set_xlim(global_min, global_max)
    #     ax2.set_xlim(global_min, global_max)

    # Make the layout more tight
    plt.tight_layout()
    plt.show()



def plot_profiles_sorted_by_SH1950(ARGO):
    '''
    Plots the temperature and salinity profiles sorted by SH1950 values.
    '''

    sorting_indices = np.argsort(ARGO.SH1950, axis=0).flatten()
    sorted_temp_profiles = ARGO.TEMP[:,sorting_indices]
    sorted_sal_profiles = ARGO.TEMP[:,sorting_indices]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im = axs[0].imshow(sorted_temp_profiles, aspect='auto', origin='lower', 
                  interpolation='nearest', cmap=ccm.thermal)

    # Add colorbar to this axis
    cbar = fig.colorbar(im, ax=axs[0])
    # Set text for the colorbar
    cbar.ax.set_ylabel('Temperature (°C)', labelpad=10)

    axs[0].set_xlabel('Profile')
    axs[0].set_ylabel('Depth (m)')
    axs[0].set_title('Temperature Profiles Sorted by SH1950')
    axs[0].invert_yaxis()

    im2 = axs[1].imshow(sorted_sal_profiles, aspect='auto', origin='lower', 
                  interpolation='nearest', cmap=ccm.haline)
    cbar = fig.colorbar(im2, ax=axs[1])
    # Set text for the colorbar
    cbar.ax.set_ylabel('Salinity (PSU)', labelpad=10)
    axs[1].set_xlabel('Profile')
    axs[1].set_ylabel('Depth (m)')
    axs[1].set_title('Salinity Profiles Sorted by SH1950')
    axs[1].invert_yaxis()
    plt.show()