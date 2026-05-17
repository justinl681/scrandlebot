import logging
import pathlib
from io import BytesIO
import pandas as pd
from PIL import Image
from pyrate_limiter import limiter_factory
from pyrate_limiter.extras.requests_limiter import RateLimitedRequestsSession

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger.setLevel(logging.DEBUG)

limiter = limiter_factory.create_inmemory_limiter(rate_per_duration=10)
session = RateLimitedRequestsSession(limiter)


def flatten(data):
    images = []
    for comparison in data:
        images.append(comparison[0])
        images.append(comparison[1])

    return images


data_dir = pathlib.Path(__file__).parent.parent / "data"
desired_images = 100_000
count = 0

try:
    df = pd.read_csv('../images.csv')
except FileNotFoundError:
    df = pd.DataFrame({
        "id": [],
        "filename": [],
        "title": [],
        "score": []
    })

url = "https://scrandle.com/practice"
initial = count

try:
    while count < desired_images + initial:
        r = session.get("https://scrandle.com/practice")
        r.raise_for_status()

        for image in flatten(r.json()):
            if image["id"] in df["id"].values:
                logging.debug(f"Image {count} already exists...")
                count += 1
                continue
            image_name = data_dir / f"images{count}.jpeg"
            im_data = session.get(image["images_new"][0])
            with Image.open(BytesIO(im_data.content)) as im:
                im_resized = im.resize((256, 256))
                im_resized.save(image_name, "JPEG")

            data = pd.DataFrame({
                "id": [image["id"]],
                "filename": [f"data/images{count}.jpeg"],
                "title": [image["title"]],
                "score": [round(image["rating"] / 100, 3)]
            })

            df = pd.concat([df, data])

            count += 1

except KeyboardInterrupt:
    logging.info(f"Keyboard interrupt... Stopping at {count + 1} images.")
finally:
    session.close()
    df.to_csv('images.csv', index=False, encoding='utf-8')
