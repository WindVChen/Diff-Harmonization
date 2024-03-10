import google.generativeai as genai
import PIL.Image as Image
import glob
from natsort import natsorted, ns
import os

os.environ['http_proxy'] = 'http://127.0.0.1:7890'

images_root = 'demo/composite'
masks_root = 'demo/mask'
new_caption_path = 'demo/caption_multi2.txt'  # where to save the caption.
prompt_num = 2  # How many different prompts for one image.
api_key = "Your KEY"

"""
Generate prompts with Google Genmini. Please refer to Section 3.1 `Imaging Condition Description Generation` in our 
paper-v2 for more details. 

Note that in case network connection issues, we here provide an offline generation script. That is, pre-generated multiple
prompts in advance. This will not hurt performance too much. You can also easily modify and incorporate this script 
into the harmonization pipeline to achieve an online generation as presented in Section 3.3 in our paper-v2.
"""


def get_caption(image_path, mask_path):
    try:

        genai.configure(api_key=api_key)

        # Set up the model
        generation_config = {
            "temperature": 0.3,
            "top_p": 1,
            "top_k": 64,
            "max_output_tokens": 4096,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]

        model = genai.GenerativeModel(model_name="gemini-pro-vision",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        prompt_parts = [
            "I want to choose some words to describe the composite image, which is made by superimposing cut-out object onto the background image."
        ]
        prompt_parts.append(
            "This is the provided image:")
        img = Image.open(image_path).convert('RGB')
        # resize
        img = img.resize((1024, int(1024 * img.height / img.width)), Image.ANTIALIAS)

        prompt_parts.append(img)

        prompt_parts.append("This is the mask of the foreground object:")
        mask_img = Image.open(mask_path).convert('L')
        # resize
        mask_img = mask_img.resize((1024, int(1024 * img.height / img.width)), Image.ANTIALIAS)
        prompt_parts.append(mask_img)

        prompt_parts.append(
            "The foreground region is the mask region, while the rest constitutes the background.\n"
            "Here, I provide a set of descriptive words categorized in a dictionary as follows: \n"  # Imaging description：sunset,sunrise,night,daytime,day,dusk,dawn,evening,morning
            "{'brightness':[dazzling, bright, dim, dull, shaded, shadowed],"
            "'weather':[cloudy, sunny, rainy, snowy, foggy, windy, stormy, clear, misty],"
            "'temperature':[hot, warm, cool, cold, icy],"
            "'season':[spring, summer, autumn, winter],"
            "'time':[dawn, sunrise, daylight, twilight, sunset, dusk, dark, night],"
            "'color tone':[greyscale, neon, golden, white, blue, green, yellow, orange, red, earthy],"
            "'environment':[city, rural, lake, ocean, mountain, forest, desert, grassland, sky, space, indoor, street]}\n",
        )
        prompt_parts.append(
            "Now, I need to first give the name of the foreground object and then select appropriate words from the above dictionary to describe both the foreground object and background. Here are the specific instructions: \n "
            "1. Describe the name of the foreground object\n"
            "2. Choose one or more words from the entire dictionary that best describe the style of foreground. (e.g. brightness, color tone...)\n"
            "3. Choose one or more words from the entire dictionary that best describe the background. (e.g. brightness, weather, temperature, season ...)\n"
            "Note: Choose only one word from each list and ensure that a word from the 'brightness' list is included in the selection.\n"
            "Note: The output format should be: (foreground object name) X X ... & (foreground) X X ... & (background) X X ..., where X represents a word. "
            "For example, dog & (foreground) bright summer & (background) winter dull greyscale. \n "
            # "Note: Please refer to the provided image for the actual chosen words. \n" (foreground) bright & (background) summer dim blue, or,
            "Ensure adherence to this format in your response; any other formats will not be accepted."
        )

        response = model.generate_content(prompt_parts, stream=True)
        response.resolve()

    except Exception as e:
        print(e)
        print("get caption failed, try again.....")
        return ''

    print(response.text)
    return response.text.strip(' ')


composite_images = []
mask_images = []
for i in glob.glob(os.path.join(images_root, "*")):
    composite_images.append(i)
for i in glob.glob(os.path.join(masks_root, "*")):
    mask_images.append(i)
composite_images = natsorted(composite_images, alg=ns.PATH)
mask_images = natsorted(mask_images, alg=ns.PATH)

for i in range(0, len(composite_images)):
    image_path = composite_images[i]
    mask_path = mask_images[i]
    caption = ''
    for t in range(prompt_num):
        while len(caption.split('&')) != 3 or "(foreground)" not in caption or "(background)" not in caption:
            caption = get_caption(image_path, mask_path)
        object_name = caption.split('&')[0].strip()
        new_caption = "{} {},{} {}".format(object_name, caption.split('&')[1].split(')')[1].strip(),
                                           object_name, caption.split('&')[2].split(')')[1].strip())
        print("new caption: ", new_caption)
        caption = ''
        with open(new_caption_path, 'a') as f:
            f.write(new_caption)
            if t != 1:
                f.write(';')
    with open(new_caption_path, 'a') as f:
        f.write('\n')
