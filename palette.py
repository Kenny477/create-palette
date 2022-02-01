import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np

class Palette:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    
    def __init__(self, image, clusters: int = 3, downsize: float = 1.00):
        img = Image.open(image)
        self.WIDTH = img.size[0]
        self.HEIGHT = img.size[1]
        img = img.resize((round(img.size[0]*downsize), round(img.size[1]*downsize)), Image.LANCZOS)
        img.save('img1-downsized.png')
        self.IMAGE = img
        self.CLUSTERS = clusters

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def get_size(self):
        return (self.WIDTH, self.HEIGHT)

    # TODO: Work on filtering out neutrals/unwanted colors
    def pixelFilter(self, rgb):
        return True

    def dominantColors(self, neutral=True):
        #read image
        img = np.array(self.IMAGE)

        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        img = img if neutral else [rgb for rgb in img if self.pixelFilter(rgb)]

        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS, random_state=477)
        kmeans.fit(img)
        
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        
        #save labels
        self.LABELS = kmeans.labels_
        
        #returning after converting to integer from float
        return self.COLORS.astype(int)
    
    def plotClusters(self, save=''):
        #plotting 
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        img = np.array(self.IMAGE)
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        for label, pix in zip(self.LABELS, img):
            ax.scatter(pix[0], pix[1], pix[2], color = self.rgb_to_hex(self.COLORS[label]))
        if save:
            plt.savefig(save)
        else:
            plt.show()
    
    def drawPalette(self, size: int = 100, save='') -> Image.Image:
        numLabels = np.arange(0, self.CLUSTERS+1)
         
        (hist, _) = np.histogram(self.LABELS, bins = numLabels)
        hist = hist.astype(float)
        hist /= hist.sum()
        
        colors = self.COLORS

        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()] 
        
        colors = colors.astype(int)

        img = Image.new("RGB", (size*self.CLUSTERS, size))

        for index in range(self.CLUSTERS):
            for i in range(size):
                for j in range(size):
                    img.putpixel((i+size*index, j), tuple(colors[index]))

        if save:
            img.save(save)
        return img

def create_palette(image, clusters=10) -> tuple:
    p = Palette(image, clusters, downsize=0.1) 
    colors = p.dominantColors(neutral=False)
    colors = [p.rgb_to_hex(color) for color in colors]
    size = p.get_size()
    return (p.drawPalette(100), colors, size[0], size[1])

