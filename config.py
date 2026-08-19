import warnings
warnings.filterwarnings('ignore')

import configparser

def read_config():
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


if __name__ == "__main__":
    # Call the function to read the configuration file
    config_data = read_config()

    TARGET_NAME = config_data['target']
    PERIOD = float(config_data['period'])
    IS_PHASE = config_data['is_phase']
    PHASE_ZERO = float(config_data['phase_zero'])
    REFERENCE_IMAGE = config_data['reference_image']
    STARS = config_data['stars']
    TITLE = TARGET_NAME + "_" + DATE
    if IS_PHASE == False:
        PHASE_ZERO = None
    print("Config file read.")


STARS1 = STARS.replace("[", "")
STARS2 = STARS1.replace("]", "")
STARS3 = STARS2.replace(" ", "")
STARS4 = STARS3.replace("'", "")
STARS5 = STARS4.split(",")
STAR_COORDS = star_list_pix = [(float(STARS5[0]), float(STARS5[1])), (float(STARS5[2]), float(STARS5[3])), (float(STARS5[4]), float(STARS5[5])), (float(STARS5[6]), float(STARS5[7])), (float(STARS5[8]), float(STARS5[9]))]