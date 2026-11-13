"""Dataset locations. All public, no account. Verified 2026-11-13 (see docs/DECISIONS.md).

If a URL stops working, fix it here, note it in DECISIONS.md, and rerun `landscore fetch`.
"""

from __future__ import annotations

# --- County boundaries: US Census cartographic boundary file, 1:500k, 2023 (11.6 MB zip).
COUNTIES_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"

# --- Annual NLCD Collection 1.1 land cover, 1985-2024, CONUS COG per year (~1.4 GB each;
# we read only the county window over HTTP range requests).
# The official USGS bucket (s3://usgs-landcover) is requester-pays, so we read a public
# mirror of the same files on source.coop. Fallback: MRLC's own WCS (official, slower).
NLCD_MIRROR = (
    "https://data.source.coop/kylebarron/usgs-landcover/annual-nlcd/c1/v1/cu/mosaic/"
    "Annual_NLCD_LndCov_{year}_CU_C1V1.tif"
)
NLCD_WCS = (
    "https://www.mrlc.gov/geoserver/mrlc_Land-Cover-Native_conus_year_data/wcs"
    "?service=WCS&version=1.0.0&request=GetCoverage"
    "&coverage=mrlc_Land-Cover-Native_conus_year_data:Land-Cover-Native_conus_year_data"
    "&CRS=EPSG:5070&BBOX={xmin},{ymin},{xmax},{ymax}&RESX=30&RESY=30&FORMAT=GeoTIFF"
    "&TIME={year}-01-01T00:00:00.000Z"
)
NLCD_NODATA = 250

# --- USFS Tree Canopy Cover, "Science" product, v2025-6, 1985-2025. One ~6 GB zip per year
# holding an uncompressed COG, so GDAL can read just the county window through the zip.
TCC_ZIP = (
    "https://data.fs.usda.gov/geodata/rastergateway/treecanopycover/docs/v2025-6/"
    "science_tcc_conus_{year}_v2025-6_wgs84.zip"
)
TCC_MEMBER = "science_tcc_conus_wgs84_v2025-6_{year}0101_{year}1231.tif"
TCC_NODATA = 255

# --- Elevation: USGS 3DEP 1/3 arc-second seamless DEM, 1x1 degree COG tiles.
DEM_TILE = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/"
    "{tile}/USGS_13_{tile}.tif"
)

# --- Soils: gNATSGO map unit key raster (10 m) from Microsoft Planetary Computer
# (anonymous, signed URLs), tables from USDA Soil Data Access (current SSURGO).
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
GNATSGO_COLLECTION = "gnatsgo-rasters"
SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"
SEEDLING_MORTALITY_RULE = "FOR - Potential Seedling Mortality"

# --- Climate: NOAA nClimGrid 1991-2020 annual normals (~5 km) on Planetary Computer.
CLIMATE_COLLECTION = "noaa-climate-normals-gridded"
CLIMATE_ITEM = "1991_2020-annual"
