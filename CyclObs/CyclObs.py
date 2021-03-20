# # %%
import pandas as pd


class CyclObs:
    base_url = "https://cyclobs.ifremer.fr/app/api/getData"

    def __init__(self, include_cols=True, cyclone_name=None, sid=None, instrument=None, mission=None, basin=None, acquisition_start_time=None, acquisition_stop_time=None, cat_min=None, cat_max=None, product_type=None, nopath=None, noheader=None):
        """
        include_cols: bool, default True
            comma separated list to format the csv-like output to include the given columns.
            Options are: cyclone_name, sid, data_url, acquisition_start_time, cyclone_start_date, cyclone_stop_date, instrument, mission, vmax, maximum_cyclone_category, basin.
            Defaults to cyclone_name, data_url.

        cyclone_name: str
            commma separated list to filter wanted cyclones.
            Defaults to all cyclones.

        sid: str
            commma separated list to filter wanted storm id.
            Defaults to all storm ids.

        instrument: str
            commma separated list to filter wanted instruments.
            Defaults to all instruments.
            To see available values go to https://cyclobs.ifremer.fr/app/api/allInstruments?short

        mission: str
            comma separated list to filter wanted missions. Defaults to all missions.
            To see available values go to https://cyclobs.ifremer.fr/app/api/allMissions?short

        basin: str
            commma separated list to filter wanted basins. Defaults to all basins.
            To see available values go to https://cyclobs.ifremer.fr/app/api/allBasins

        acquisition_start_time: str
            returned acquisitions returned will have acquisition start time above or equal to startdate.
            Format : YYYY-MM-DD. Defaults to no time limit.

        acquisition_stop_time: str
            returned acquisitions returned will have acquisition stop time below or equal to stopdate.
            Format : YYYY-MM-DD. Defaults to no time limit

        cat_min: str
            minimum category (including the cat_min given limit) wanted for cyclone's acquisitions.
            Can be : dep, storm or cat-X with X from 1 to 5. Defaults to no category lower limit.

        cat_max: str
            maximum category (excluding the cat_max given limit) wanted for cyclone's acquisitions.
            Can be : dep, storm or cat-X with X from 1 to 5. Defaults to no category higher limit.
            cat_max must be above cat_min.

        product_type: str
            product type choice, either 'swath' or 'gridded'.
            SMOS and SMAP data are only available in gridded format.
            SAR data are available both in swath and gridded formats.

        nopath: bool
            if set (no value needed) only the filenames will be returned in the column data_url

        noheader: bool
            if set (no value needed) the csv header line will not be set in the ouput
        """
        
        
