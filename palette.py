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
        img = img.resize((round(img.size[0]*downsize), round(img.size[1]*downsize)), Image.LANCZOS)
        self.IMAGE = img
        self.CLUSTERS = clusters
    
    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def dominantColors(self, neutral=True):
        #read image
        img = np.array(self.IMAGE)

        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))

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
        else:
            img.show()


img = 'img1.jpg'
clusters = 10

p = Palette(img, clusters, downsize=0.01) 
colors = p.dominantColors(neutral=False)
p.plotClusters(save='img1-clusters.png')
p.drawPalette(100, save='img1-palette.png')
print(colors)

