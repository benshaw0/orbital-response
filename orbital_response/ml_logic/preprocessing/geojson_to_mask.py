import json
import geopandas as gpd
import rasterio
from shapely import wkt
from rasterio.features import rasterize
from shapely.geometry import box

DAMAGE_MAPPING = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4
}

def load_custom_json(json_path):
    """
    Load Geometries from JSON (xView2 format).
    returns the lists for those geos (geometry, class_id).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    geometries = []
    for item in data["features"]["lng_lat"]:
        try:
            geom = wkt.loads(item["wkt"])
            subtype = item["properties"].get("subtype", "no-damage")
            class_id = DAMAGE_MAPPING.get(subtype, 0)
            geometries.append((geom, class_id))
        except Exception as e:
            print(f"❌ Error in geometry: {e}")
            continue

    return geometries

def geojson_to_mask(json_path, image_path, output_shape):
    """
    Converts an annotated JSON file into a raster mask.
    Uses the original .tif transform to ensure perfect alignment.
    """
    shapes = load_custom_json(json_path)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        [{"class_id": c} for _, c in shapes],
        geometry=[g for g, _ in shapes],
        crs="EPSG:4326"
    )

    with rasterio.open(image_path) as src:
        height, width = output_shape
        bounds = src.bounds
        crs_img = src.crs
        transform = src.transform  #USING ORIGNINAL TRANSFORM

    # Reproyectar al CRS de la imagen
    gdf = gdf.to_crs(crs_img)

    # Recorte a los límites de la imagen
    gdf = gdf.clip(box(*bounds))
    gdf = gdf[gdf.is_valid & gdf.geometry.notnull() & gdf.geometry.area > 0]

    # Rasterización usando el transform original
    mask = rasterize(
        ((geom, row.class_id) for geom, row in zip(gdf.geometry, gdf.itertuples())),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    return mask
