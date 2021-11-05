from PIL import Image
from PIL import ImageFilter
import pandas as pd
import numpy as np
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
from matplotlib import pyplot as plt

SAMPLE_SIZE = 10
PALETTE_SIZE = 10

def clamp(x: int) -> int:
    return max(0, min(x, 255))

def rgbtohex(rgb: tuple) -> str:
    (r, g, b) = rgb
    return "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))

def getMax(array: list, n: int) -> list:
    return sorted(array, reverse=True)[:n]

def createHistogram(image: Image):
    hist = img.histogram()
    np_hist, bin_edges = np.histogram(hist, bins=256)

    plt.figure()
    plt.title("Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")

    plt.plot(bin_edges[0:-1], np_hist)  # <- or here
    plt.savefig('histogram.png')

def drawPalette(palette: list, size: int) -> Image.Image:
    img = Image.new("RGB", (size*len(palette), size))
    for index, color in enumerate(palette):
        for i in range(size):
            for j in range(size):
                img.putpixel((i+size*index, j), color)
    return img

def findDominantColors(img: Image) -> list:
    r = []
    g = []
    b = []
    for i in range(img.height):
        for j in range(img.width):
            r.append(img.getpixel((j,i))[0])
            g.append(img.getpixel((j,i))[1])
            b.append(img.getpixel((j,i))[2])
        
    df = pd.DataFrame({'red' : r, 'green' : g, 'blue' : b})
    
    df['scaled_color_red'] = whiten(df['red'])
    df['scaled_color_blue'] = whiten(df['blue'])
    df['scaled_color_green'] = whiten(df['green'])
    
    cluster_centers, _ = kmeans(df[['scaled_color_red',
                                    'scaled_color_blue',
                                    'scaled_color_green']], PALETTE_SIZE)
    
    dominant_colors = []
    
    red_std, green_std, blue_std = df[['red',
                                        'green',
                                        'blue']].std()
    
    for cluster_center in cluster_centers:
        red_scaled, green_scaled, blue_scaled = cluster_center
        dominant_colors.append((
            red_scaled * red_std / 255,
            green_scaled * green_std / 255,
            blue_scaled * blue_std / 255
        ))
    plt.imshow([dominant_colors])
    plt.show()
    return dominant_colors

# f1 = img.filter(ImageFilter.FIND_EDGES)
# f2 = img.filter(ImageFilter.EMBOSS)
# colors = img.getcolors(img.size[0] * img.size[1])
# df = pd.DataFrame(colors, columns=['count', 'color'])
# df['sampled'] = df['color'].apply(lambda x: tuple(int(value/SAMPLE_SIZE) for value in x))
# df = df.groupby(by=['sampled'], dropna=True).sum()
# df.index = df.index.map(lambda x: (tuple(value * 5 for value in x)))
# df = df.sort_values(by='count', ascending=False)
# top = list(df.head(PALETTE_SIZE).index)
# drawPalette(top, 100).save('img1-palette.png')

img = Image.open('img1.jpg')
print(findDominantColors(img))



