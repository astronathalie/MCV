import warnings
warnings.filterwarnings('ignore')

import configparser

def read_config(config_file):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(config_file)

    # Access values from the configuration file
    target_name = config.get('Target', 'target')
    period = config.getfloat('Target', 'period')
    is_phase = config.getboolean('Target', 'is phase')
    phase_zero = config.get('Target', 'phase zero')
    reference_image = config.get('Reference', 'ref image')
    stars = config.get('Reference', 'ref stars')

    # Return a dictionary with the retrieved values
    config_values = {
        'target': target_name,
        'period': period,
        'is_phase': is_phase,
        'phase_zero': phase_zero,
        'reference_image': reference_image,
        'stars': stars
    }

    return config_values


def load_config(config_file, DATE):
    config_data = read_config(config_file)

    target_name    = config_data['target']
    period         = float(config_data['period'])
    is_phase       = config_data['is_phase']
    phase_zero     = float(config_data['phase_zero'])
    reference_image = config_data['reference_image']
    stars          = config_data['stars']
    title          = target_name + "_" + DATE

    if is_phase == False:
        phase_zero = None

    # parse the stars string "[(x,y), ...]" into a list of (x, y) tuples
    nums = [float(x) for x in stars.strip("[]").replace("'", "").replace(" ", "").split(",")]
    star_coords = list(zip(nums[::2], nums[1::2]))

    print("Config file read.")

    return {
        "TARGET_NAME":     target_name,
        "PERIOD":          period,
        "IS_PHASE":        is_phase,
        "PHASE_ZERO":      phase_zero,
        "REFERENCE_IMAGE": reference_image,
        "TITLE":           title,
        "STAR_COORDS":     star_coords,
    }