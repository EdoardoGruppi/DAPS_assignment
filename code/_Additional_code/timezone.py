# Import packages
import pytz


# During the conversion we could have to consider also the differences in the daylight savings time. However, if they
# are taken into account the time required to make the conversion exponentially increases while the results remain
# almost the same.
def convert_timezones(dataframe, date_column, source_loc, target_loc):
    # Original timezone related to the date column of the dataframe
    source_time_zone = pytz.timezone(source_loc)
    # Load the target timezone
    target_time_zone = pytz.timezone(target_loc)
    for date in dataframe[date_column]:
        source_date = source_time_zone.localize(date)
        target_date = source_date.astimezone(target_time_zone)
        # Update the date column with new converted dates
        dataframe.loc[date] = target_date.strftime('%Y-%m-%d %H:%M:%S')