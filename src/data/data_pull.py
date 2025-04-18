import comtradeapicall
from ..models import (
    get_conn,
    init_int_trade_data_table,
    init_jp_trade_data_table,
    init_com_trade_data_table,
)
from tqdm import tqdm
import polars as pl
import pandas as pd
import datetime
import requests
import logging
import zipfile
import urllib3
import os


class DataPull:
    """
    This class pulls data from the CENSUS and the Puerto Rico Institute of Statistics

    """

    def __init__(
        self,
        saving_dir: str = "data/",
        database_file: str = "data.ddb",
        log_file: str = "data_process.log",
    ):
        self.saving_dir = saving_dir
        self.data_file = database_file
        self.conn = get_conn(self.data_file)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            filename=log_file,
        )
        # Check if the saving directory exists
        if not os.path.exists(self.saving_dir + "raw"):
            os.makedirs(self.saving_dir + "raw")
        if not os.path.exists(self.saving_dir + "processed"):
            os.makedirs(self.saving_dir + "processed")
        if not os.path.exists(self.saving_dir + "external"):
            os.makedirs(self.saving_dir + "external")

    def pull_int_org(self) -> None:
        """
        Pulls data from the Puerto Rico Institute of Statistics. Saves them in the
            raw directory as a parquet file.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.pull_file(
            url="http://www.estadisticas.gobierno.pr/iepr/LinkClick.aspx?fileticket=JVyYmIHqbqc%3d&tabid=284&mid=244930",
            filename=(self.saving_dir + "raw/tmp.zip"),
        )
        # Extract the zip file
        with zipfile.ZipFile(self.saving_dir + "raw/tmp.zip", "r") as zip_ref:
            zip_ref.extractall(f"{self.saving_dir}raw/")

        # Extract additional zip files
        additional_files = ["EXPORT_HTS10_ALL.zip", "IMPORT_HTS10_ALL.zip"]
        for additional_file in additional_files:
            additional_file_path = os.path.join(
                f"{self.saving_dir}raw/{additional_file}"
            )
            with zipfile.ZipFile(additional_file_path, "r") as zip_ref:
                zip_ref.extractall(os.path.join(f"{self.saving_dir}raw/"))

        # Concatenate the files
        imports = pl.scan_csv(
            self.saving_dir + "raw/IMPORT_HTS10_ALL.csv", ignore_errors=True
        )
        exports = pl.scan_csv(
            self.saving_dir + "raw/EXPORT_HTS10_ALL.csv", ignore_errors=True
        )
        pl.concat([imports, exports], how="vertical").collect().write_parquet(
            self.saving_dir + "raw/org_data.parquet"
        )

        logging.info(
            "finished extracting data from the Puerto Rico Institute of Statistics"
        )

    def insert_int_org(self) -> pl.DataFrame:
        if (
            "inttradedata"
            not in self.conn.sql("SHOW TABLES;").df().get("name").tolist()
        ):
            init_int_trade_data_table(self.data_file)
        if self.conn.sql("SELECT * FROM 'inttradedata';").df().empty:
            if not os.path.exists(f"{self.saving_dir}raw/org_data.parquet"):
                self.pull_int_org()
            if not os.path.exists(f"{self.saving_dir}external/code_agr.json"):
                logging.debug(
                    "https://raw.githubusercontent.com/ouslan/jp-imports/main/data/external/code_agr.json"
                )
                self.pull_file(
                    url="https://raw.githubusercontent.com/ouslan/jp-imports/main/data/external/code_agr.json",
                    filename=(f"{self.saving_dir}external/code_agr.json"),
                )
            agri_prod = pl.read_json(
                f"{self.saving_dir}external/code_agr.json"
            ).transpose()
            agri_prod = (
                agri_prod.with_columns(pl.nth(0).cast(pl.String).str.zfill(4))
                .to_series()
                .to_list()
            )
            int_df = pl.scan_parquet(f"{self.saving_dir}raw/org_data.parquet")
            int_df = int_df.rename(
                {col: col.lower() for col in int_df.collect_schema().names()}
            )
            int_df = int_df.with_columns(
                date=pl.col("year").cast(pl.String)
                + "-"
                + pl.col("month").cast(pl.String)
                + "-01",
                unit_1=pl.col("unit_1").str.to_lowercase(),
                unit_2=pl.col("unit_2").str.to_lowercase(),
                hts_code=pl.col("hts")
                .cast(pl.String)
                .str.zfill(10)
                .str.replace("'", ""),
                trade_id=pl.when(pl.col("import_export") == "i").then(1).otherwise(2),
            ).rename({"value": "data"})

            int_df = int_df.with_columns(pl.col("date").cast(pl.Date))

            int_df = int_df.with_columns(hs4=pl.col("hts_code").str.slice(0, 4))

            int_df = int_df.with_columns(
                agri_prod=pl.when(pl.col("hs4").is_in(agri_prod)).then(1).otherwise(0)
            )

            int_df = int_df.select(
                pl.col(
                    "date",
                    "country",
                    "trade_id",
                    "agri_prod",
                    "hts_code",
                    "hts_desc",
                    "data",
                    "qty_1",
                    "unit_1",
                    "qty_2",
                    "unit_2",
                )
            ).collect()

            self.conn.sql("INSERT INTO 'inttradedata' BY NAME SELECT * FROM int_df;")
            logging.info("finished inserting data into the database")
            return self.conn.sql("SELECT * FROM 'inttradedata';").pl()
        else:
            return self.conn.sql("SELECT * FROM 'inttradedata';").pl()

    def pull_int_jp(self, update: bool = False) -> None:
        """
        Pulls data from the Puerto Rico Institute of Statistics used by the JP.
            Saved them in the raw directory as parquet files.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not os.path.exists(self.saving_dir + "external/code_classification.json"):
            logging.debug(
                "pull file from https://raw.githubusercontent.com/ouslan/jp-imports/main/data/external/code_classification.json"
            )
            self.pull_file(
                url="https://raw.githubusercontent.com/ouslan/jp-imports/main/data/external/code_classification.json",
                filename=(self.saving_dir + "external/code_classification.json"),
            )

        if not os.path.exists(self.saving_dir + "raw/jp_data.parquet") or update:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logging.debug(
                "https://datos.estadisticas.pr/dataset/027ddbe1-c51c-46bf-aec3-a62d5d7e8539/resource/b8367825-a3de-41cf-8794-e42c10987b6f/download/ftrade_all_iepr.csv"
            )
            url = "https://datos.estadisticas.pr/dataset/027ddbe1-c51c-46bf-aec3-a62d5d7e8539/resource/b8367825-a3de-41cf-8794-e42c10987b6f/download/ftrade_all_iepr.csv"
            self.pull_file(
                url=url, filename=(self.saving_dir + "raw/jp_data.csv"), verify=False
            )
            pl.read_csv(
                f"{self.saving_dir}/raw/jp_data.csv", ignore_errors=True
            ).write_parquet(f"{self.saving_dir}/raw/jp_data.parquet")

        logging.info("Pulling data from the Puerto Rico Institute of Statistics")

    def insert_int_jp(self) -> pl.DataFrame:
        if "jptradedata" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            init_jp_trade_data_table(self.data_file)

        if self.conn.sql("SELECT * FROM 'jptradedata';").df().empty:
            if not os.path.exists(f"{self.saving_dir}raw/jp_data.parquet"):
                self.pull_int_jp()
            if not os.path.exists(f"{self.saving_dir}external/code_agr.json"):
                logging.debug(
                    "https://raw.githubusercontent.com/ouslan/jp-imports/main/data/external/code_agr.json"
                )
                self.pull_file(
                    url="https://raw.githubusercontent.com/ouslan/jp-imports/main/data/external/code_agr.json",
                    filename=(f"{self.saving_dir}external/code_agr.json"),
                )
            agri_prod = pl.read_json(
                f"{self.saving_dir}external/code_agr.json"
            ).transpose()
            agri_prod = (
                agri_prod.with_columns(pl.nth(0).cast(pl.String).str.zfill(4))
                .to_series()
                .to_list()
            )
            jp_df = pl.read_parquet(f"{self.saving_dir}raw/jp_data.parquet")
            jp_df = jp_df.rename(
                {col: col.lower() for col in jp_df.collect_schema().names()}
            )
            jp_df = jp_df.with_columns(
                date=pl.col("year").cast(pl.String)
                + "-"
                + pl.col("month").cast(pl.String)
                + "-01",
                unit_1=pl.col("unit_1").str.to_lowercase(),
                unit_2=pl.col("unit_2").str.to_lowercase(),
                hts_code=pl.col("commodity_code")
                .cast(pl.String)
                .str.zfill(10)
                .str.replace("'", ""),
                trade_id=pl.when(pl.col("trade") == "i").then(1).otherwise(2),
            )

            jp_df = jp_df.with_columns(pl.col("date").cast(pl.Date))

            jp_df = jp_df.with_columns(
                agri_prod=pl.when(pl.col("hts_code").is_in(agri_prod))
                .then(1)
                .otherwise(0)
            )
            jp_df = jp_df.with_columns(
                sitc=pl.when(pl.col("sitc_short_desc").str.starts_with("Civilian"))
                .then(9998)
                .when(pl.col("sitc_short_desc").str.starts_with("-"))
                .then(9999)
                .otherwise(pl.col("sitc"))
            )
            jp_df = jp_df.filter(pl.col("hts_code").is_not_null())
            jp_df = jp_df.select(
                pl.col(
                    "date",
                    "country",
                    "trade_id",
                    "agri_prod",
                    "hts_code",
                    "hts_desc",
                    "data",
                    "qty_1",
                    "unit_1",
                    "qty_2",
                    "unit_2",
                    "sitc",
                    "naics",
                )
            )
            self.conn.sql("INSERT INTO 'jptradedata' BY NAME SELECT * FROM jp_df;")
            logging.info("finished inserting data into the database")
            return self.conn.sql("SELECT * FROM 'jptradedata';").pl()
        else:
            return self.conn.sql("SELECT * FROM 'jptradedata';").pl()

    def pull_comtrade(self, iso: str, trade_id, date, code) -> pl.DataFrame:
        df = comtradeapicall.previewFinalData(
            typeCode="C",
            freqCode="M",
            clCode="HS",
            period=date,
            reporterCode=None,
            cmdCode=code,
            flowCode=trade_id,
            partnerCode=iso,
            partner2Code=None,
            customsCode=None,
            motCode=None,
            maxRecords=500,
            format_output="JSON",
            aggregateBy=None,
            breakdownMode="classic",
            countOnly=None,
            includeDesc=True,
        )
        if df.empty:
            return pl.DataFrame(df)

        df = pl.from_pandas(df).cast(pl.String)
        return pl.DataFrame(df)

    def insert_comtrade(self, iso: str):
        if (
            "comtradetable"
            not in self.conn.sql("SHOW TABLES;").df().get("name").tolist()
        ):
            init_com_trade_data_table(self.data_file)

        codes = (
            self.insert_int_org()
            .select(pl.col("hts_code").str.slice(0, 2).unique())
            .to_series()
            .to_list()
        )
        for year in range(2010, datetime.date.today().year + 1):
            for month in range(1, 13):
                if year == datetime.date.today().year and month >= datetime.date.today().month:
                    continue
                for code in codes:
                    if (
                        not self.conn.sql(
                            f"SELECT * FROM 'comtradetable' WHERE refYear={year} AND refMonth={month} AND cmdCode={code} AND partnerCode={iso} LIMIT(1);"
                        )
                        .df()
                        .empty
                    ):
                        continue
                    df = self.pull_comtrade(
                        iso, "X", f"{year}{str(month).zfill(2)}", code
                    )
                    if df.is_empty():
                        dummy_df = pl.DataFrame(
                            [
                                pl.Series("typeCode", [""], dtype=pl.String),
                                pl.Series("freqCode", [""], dtype=pl.String),
                                pl.Series("refPeriodId", [""], dtype=pl.String),
                                pl.Series("refYear", [str(year)], dtype=pl.String),
                                pl.Series("refMonth", [str(month)], dtype=pl.String),
                                pl.Series("period", [""], dtype=pl.String),
                                pl.Series("reporterCode", [""], dtype=pl.String),
                                pl.Series("reporterISO", [""], dtype=pl.String),
                                pl.Series("reporterDesc", [""], dtype=pl.String),
                                pl.Series("flowCode", [""], dtype=pl.String),
                                pl.Series("flowDesc", [""], dtype=pl.String),
                                pl.Series("partnerCode", [iso], dtype=pl.String),
                                pl.Series("partnerISO", [""], dtype=pl.String),
                                pl.Series("partnerDesc", [""], dtype=pl.String),
                                pl.Series("partner2Code", [""], dtype=pl.String),
                                pl.Series("partner2ISO", [""], dtype=pl.String),
                                pl.Series("partner2Desc", [""], dtype=pl.String),
                                pl.Series("classificationCode", [""], dtype=pl.String),
                                pl.Series(
                                    "classificationSearchCode", [""], dtype=pl.String
                                ),
                                pl.Series(
                                    "isOriginalClassification",
                                    [""],
                                    dtype=pl.String,
                                ),
                                pl.Series("cmdCode", [code], dtype=pl.String),
                                pl.Series("cmdDesc", [""], dtype=pl.String),
                                pl.Series("aggrLevel", [""], dtype=pl.String),
                                pl.Series("isLeaf", [""], dtype=pl.String),
                                pl.Series("customsCode", [""], dtype=pl.String),
                                pl.Series("customsDesc", [""], dtype=pl.String),
                                pl.Series("mosCode", [""], dtype=pl.String),
                                pl.Series("motCode", [""], dtype=pl.String),
                                pl.Series("motDesc", [""], dtype=pl.String),
                                pl.Series("qtyUnitCode", [""], dtype=pl.String),
                                pl.Series("qtyUnitAbbr", [""], dtype=pl.String),
                                pl.Series("qty", [0.0], dtype=pl.Float64),
                                pl.Series("isQtyEstimated", [""], dtype=pl.String),
                                pl.Series("altQtyUnitCode", [""], dtype=pl.String),
                                pl.Series("altQtyUnitAbbr", [""], dtype=pl.String),
                                pl.Series("altQty", [0.0], dtype=pl.Float64),
                                pl.Series("isAltQtyEstimated", [""], dtype=pl.String),
                                pl.Series("netWgt", [0.0], dtype=pl.Float64),
                                pl.Series("isNetWgtEstimated", [""], dtype=pl.String),
                                pl.Series("grossWgt", [0.0], dtype=pl.Float64),
                                pl.Series("isGrossWgtEstimated", [""], dtype=pl.String),
                                pl.Series("cifvalue", [0.0], dtype=pl.Float64),
                                pl.Series("fobvalue", [0.0], dtype=pl.Float64),
                                pl.Series("primaryValue", [0.0], dtype=pl.Float64),
                                pl.Series(
                                    "legacyEstimationFlag", [""], dtype=pl.String
                                ),
                                pl.Series("isReported", [""], dtype=pl.String),
                                pl.Series("isAggregate", [""], dtype=pl.String),
                            ]
                        )
                        self.conn.sql(
                            "INSERT INTO 'comtradetable' BY NAME SELECT * FROM dummy_df;"
                        )

                        logging.warning(
                            f"Returned None for {year}-{month} for {code} Inserted dummy records for iso {iso}"
                        )
                        continue
                    elif len(df) == 500:
                        logging.critical(
                            f"Error: {year}-{month} {code} and iso {iso} returned 500 rows."
                        )

                    self.conn.sql(
                        "INSERT INTO 'comtradetable' BY NAME SELECT * FROM df;"
                    )
                    logging.info(
                        f"Succesfully inserted {len(df)} records for {year}-{month} for {code} for iso {iso}"
                    )
        return self.conn.sql("SELECT * FROM 'comtradetable';").pl()

    def pull_census_hts(
        self, end_year: int, start_year: int, exports: bool, state: str
    ) -> None:
        """
        Pulls HTS data from the Census and saves them in a parquet file.

        Parameters
        ----------
        end_year: int
            The last year to pull data from.
        start_year: int
            The first year to pull data from.
        exports: bool
            If True, pulls exports data. If False, pulls imports data.
        state: str
            The state to pull data from (e.g. "PR" for Puerto Rico).

        Returns
        -------
        None
        """

        empty_df = [
            pl.Series("date", dtype=pl.Datetime),
            pl.Series("census_value", dtype=pl.Int64),
            pl.Series("comm_level", dtype=pl.String),
            pl.Series("commodity", dtype=pl.String),
            pl.Series("country_name", dtype=pl.String),
            pl.Series("contry_code", dtype=pl.String),
        ]
        census_df = pl.DataFrame(empty_df)
        base_url = "https://api.census.gov/data/timeseries/"
        key = os.getenv("CENSUS_API_KEY")

        if exports:
            param = "CTY_CODE,CTY_NAME,ALL_VAL_MO,COMM_LVL,E_COMMODITY"
            flow = "intltrade/exports/statehs"
            naming = {
                "CTY_CODE": "contry_code",
                "CTY_NAME": "country_name",
                "ALL_VAL_MO": "census_value",
                "COMM_LVL": "comm_level",
                "E_COMMODITY": "commodity",
            }
            saving_path = f"{self.saving_dir}/raw/census_hts_exports.parquet"
        else:
            param = "CTY_CODE,CTY_NAME,GEN_VAL_MO,COMM_LVL,I_COMMODITY"
            flow = "intltrade/imports/statehs"
            naming = {
                "CTY_CODE": "contry_code",
                "CTY_NAME": "country_name",
                "GEN_VAL_MO": "census_value",
                "COMM_LVL": "comm_level",
                "I_COMMODITY": "commodity",
            }
            saving_path = f"{self.saving_dir}/raw/census_hts_imports.parquet"

        for year in range(start_year, end_year + 1):
            url = f"{base_url}{flow}?get={param}&STATE={state}&key={key}&time={year}"
            r = requests.get(url).json()
            df = pl.DataFrame(r)
            names = df.select(pl.col("column_0")).transpose()
            df = df.drop("column_0").transpose()
            df = df.rename(names.to_dicts().pop()).rename(naming)
            df = df.with_columns(
                date=(pl.col("time") + "-01").str.to_datetime("%Y-%m-%d")
            )
            df = df.select(
                pl.col(
                    "date",
                    "census_value",
                    "comm_level",
                    "commodity",
                    "country_name",
                    "contry_code",
                )
            )
            df = df.with_columns(pl.col("census_value").cast(pl.Int64))
            census_df = pl.concat([census_df, df], how="vertical")

        census_df.write_parquet(saving_path)

    def pull_census_naics(
        self, end_year: int, start_year: int, exports: bool, state: str
    ) -> None:
        """
        Pulls NAICS data from the Census and saves them in a parquet file.

        Parameters
        ----------
        end_year: int
            The last year to pull data from.
        start_year: int
            The first year to pull data from.
        exports: bool
            If True, pulls exports data. If False, pulls imports data.
        state: str
            The state to pull data from (e.g. "PR" for Puerto Rico).

        Returns
        -------
        None
        """
        empty_df = [
            pl.Series("date", dtype=pl.Datetime),
            pl.Series("census_value", dtype=pl.Int64),
            pl.Series("comm_level", dtype=pl.String),
            pl.Series("naics_code", dtype=pl.String),
            pl.Series("country_name", dtype=pl.String),
            pl.Series("contry_code", dtype=pl.String),
        ]
        census_df = pl.DataFrame(empty_df)
        base_url = "https://api.census.gov/data/timeseries/"
        key = os.getenv("CENSUS_API_KEY")

        if exports:
            param = "CTY_CODE,CTY_NAME,ALL_VAL_MO,COMM_LVL,NAICS"
            flow = "intltrade/exports/statenaics"
            naming = {
                "CTY_CODE": "contry_code",
                "CTY_NAME": "country_name",
                "ALL_VAL_MO": "census_value",
                "COMM_LVL": "comm_level",
                "NAICS": "naics_code",
            }
            saving_path = f"{self.saving_dir}/raw/census_naics_exports.parquet"
        else:
            param = "CTY_CODE,CTY_NAME,GEN_VAL_MO,COMM_LVL,NAICS"
            flow = "intltrade/imports/statenaics"
            naming = {
                "CTY_CODE": "contry_code",
                "CTY_NAME": "country_name",
                "GEN_VAL_MO": "census_value",
                "COMM_LVL": "comm_level",
                "NAICS": "naics_code",
            }
            saving_path = f"{self.saving_dir}/raw/census_naics_imports.parquet"

        for year in range(2010, datetime.date.today().year + 1):
            url = f"{base_url}{flow}?get={param}&STATE={state}&key={key}&time={year}"
            r = requests.get(url).json()
            df = pl.DataFrame(r)
            names = df.select(pl.col("column_0")).transpose()
            df = df.drop("column_0").transpose()
            df = df.rename(names.to_dicts().pop()).rename(naming)
            df = df.with_columns(
                date=(pl.col("time") + "-01").str.to_datetime("%Y-%m-%d")
            )
            df = df.select(
                pl.col(
                    "date",
                    "census_value",
                    "comm_level",
                    "naics_code",
                    "country_name",
                    "contry_code",
                )
            )
            df = df.with_columns(pl.col("census_value").cast(pl.Int64))
            census_df = pl.concat([census_df, df], how="vertical")

        census_df.write_parquet(saving_path)

    def pull_file(self, url: str, filename: str, verify: bool = True) -> None:
        """
        Pulls a file from a URL and saves it in the filename. Used by the class to pull external files.

        Parameters
        ----------
        url: str
            The URL to pull the file from.
        filename: str
            The filename to save the file to.
        verify: bool
            If True, verifies the SSL certificate. If False, does not verify the SSL certificate.

        Returns
        -------
        None
        """
        chunk_size = 10 * 1024 * 1024

        with requests.get(url, stream=True, verify=verify) as response:
            total_size = int(response.headers.get("content-length", 0))

            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading",
            ) as bar:
                with open(filename, "wb") as file:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            file.write(chunk)
                            bar.update(
                                len(chunk)
                            )  # Update the progress bar with the size of the chunk
