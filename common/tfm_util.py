import pytz
from pysolar.solar import get_altitude


def sanitize_name(name):
    """
    Sanitize a name replacing non-alphanumeric characters by underscores and removing accents
    :param name: The name to sanitize
    :return: The sanitized name
    """
    return ''.join(c if c.isalnum() or c in ['_', '-', '.'] else '_' for c in name) \
        .replace('á', 'a') \
        .replace('é', 'e') \
        .replace('í', 'i') \
        .replace('ó', 'o') \
        .replace('ú', 'u') \
        .replace('ñ', 'n')


def sanitize_array_names(names: list):
    """
    Sanitize an array of names
    :param names: The array of names to sanitize
    :return: The array of sanitized names
    """
    return [sanitize_name(name) for name in names]


def sanitize_column_names(df):
    """
    Sanitize the column names of a DataFrame
    :param df: The DataFrame to sanitize
    :return: The DataFrame with sanitized column names
    """
    df.columns = [sanitize_name(col) for col in df.columns]
    return df


def calculate_altitude_angle(latitude, longitude, date_time):
    """
    Calculate the altitude angle in degrees of the sun at a given location and date and time
    :param latitude: The latitude of the location
    :param longitude: The longitude of the location
    :param date_time: The date and time to calculate the altitude angle
    :return: The altitude angle in degrees
    """
    local_zone = pytz.timezone('Europe/Madrid')
    local_time = local_zone.localize(date_time)
    utc_time = local_time.astimezone(pytz.utc)
    solar_elevation = get_altitude(latitude, longitude, utc_time)
    return 0.0 if solar_elevation < 0.0 else solar_elevation
