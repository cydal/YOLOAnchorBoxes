import glob2
from plotnine import *
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image




def get_filelist(path):
  """
  Args:
      path (str): Path to text files
  Returns:
      path_lists (list): list of paths- txt files
  """
  
  return(glob2.glob(path+'**/*.txt'))


def get_tiny_class(tinytextpath, allpath):
  """
  Args:
      tinytextpath (torch.nn Model): 
      allpath (criterion) - Loss Function
  Returns:
      all_class_tiny (DataFrame): Main dataframe
      all_class_tiny_dict (Dict): class to foldername
  """

  all_classes = pd.read_csv(allpath, sep='\t', names=["class", "name"])

  tiny_text = open(tinytextpath, "r")

  content_list = tiny_text.readlines()
  content_list = "".join(content_list).split("\n")

  all_class_tiny = all_classes[all_classes["class"].isin(content_list)]

  all_class_tiny_dict = {f[1]["class"]:f[1]["name"] for f in all_class_tiny.iterrows()}

  return([all_class_tiny, all_class_tiny_dict])



def create_bbox_df(txtfiles, all_class_tiny_dict, epsilon=0.0001):
  """
  Args:
      txtfiles (list): file paths to txt files
      all_class_tiny_dict (Dict) - class to foldername
      epsilon (float): small value to prevent log 0
  Returns:
      df (DataFrame): Main Dataframe 
  """
  col_names = ["filename", "top_left_x", "top_left_y", "b_width", "b_height"]
  df = pd.concat([pd.read_csv(t, sep='\t', names=col_names) for t in txtfiles])
  df["class"] = df["filename"].str[:9].apply(lambda x: all_class_tiny_dict.get(x))

  # Center coord from top_left
  df["x"] = df["top_left_x"] + (df["b_width"] // 2)
  df["y"] = df["top_left_y"] + (df["b_height"] // 2)

  # Image Dimension
  df["img_width"] = 64
  df["img_height"] = 64

  # Norm by image height/width
  df["norm_w"] = df["img_width"] / df["img_width"]
  df["norm_h"] = df["img_height"] / df["img_height"]

  df["norm_x"] = df["x"] / df["img_width"]
  df["norm_y"] = df["y"] / df["img_height"]

  df["norm_bw"] = df["b_width"] / df["img_width"]
  df["norm_bh"] = df["b_height"] / df["img_height"]

  # Log of bbox norm 
  df["log_norm_bw"] = np.log10(df["norm_bw"] + epsilon)
  df["log_norm_bh"] = np.log10(df["norm_bh"] + epsilon)

  return(df)


def plot_scatter_bbox(df, x="norm_bw", y="norm_bh"):
  """
  Args:
      df (DataFrame): Main Dataframe 
      x (str): x_axis Column name
      y (str): y_axis Column name
  """
  print((ggplot(df, aes(x=x, y=y))
  + geom_point(colour="rebeccapurple", fill="lightskyblue")
  + labs(x=x, y=y)
  + theme_minimal()
))



def plot_class_bar(df, x="cluster", y="count"):
  """
  Args:
      df (DataFrame): Main Dataframe 
      x (str): x_axis Column name
      y (str): y_axis Column name
  """
  print((ggplot(df, aes(x=x, y=y))
  + geom_bar(colour="rebeccapurple", fill="lightskyblue", stat='identity')
  + labs(x=x, y=y)
  + theme_minimal()
))



def get_scale_factor(row):
  """
  Args:
      row (Pandas Row): DF row
  Returns:
      xy_multiplier (list): bbox width & height multiplier
  """
  anchor = kmeans.cluster_centers_[row["cluster_labels"]]
  xy_multiplier = [row["norm_bw"] / anchor[0], row["norm_bh"] / anchor[1]]
  return(xy_multiplier)



def visualize_cluster(df, x="norm_bw", y="norm_bh"):
  """
  Args:
      df (DataFrame): Main Dataframe 
      x (str): x_axis Column name
      y (str): y_axis Column name
  """
  print((ggplot(df, aes(x=x, y=y, fill="cluster_labels"))
  + geom_point()
  + labs(x=x, y=y)
  + theme_minimal()
))



def plot_anchorboxes(image, kmeans):
  """
  Args:
      image (torch.nn Model): 
      kmeans (criterion) - Loss Function
  """
  loc = f'/content/tiny-imagenet-200/train/{image["filename"][:9]}/images/{image["filename"]}'
  im = Image.open(loc)

  # Create figure and axes
  fig, ax = plt.subplots()

  # Display the image
  ax.imshow(im)

  # Create Rectangle patches

  colors = ['r', 'b', 'y', 'g', 'c', 'w']

  for i in range(len(kmeans.cluster_centers_)):
    cc = kmeans.cluster_centers_[i] * 64
    x, y = image["top_left_x"],	image["top_left_y"]
    rect = patches.Rectangle((x, y), cc[0], cc[1], linewidth=i+1, edgecolor=colors[i], facecolor='none')
  
    # Add the patch to the Axes
    ax.add_patch(rect)

  plt.show()



def compute_kmeans(df, image, k=3):
  """
  Args:
      df (DataFrame): Main Dataframe 
      image (Pandas Row) - DF row
      k (int): kmeans k param
  Returns:
      df (list): Main Dataframe Head
  """

  def get_scale_factor(row):
    anchor = kmeans.cluster_centers_[row["cluster_labels"]]
    xy_multiplier = [row["norm_bw"] / anchor[0], row["norm_bh"] / anchor[1]]
    return(xy_multiplier)

  kmeans = KMeans(n_clusters=k, random_state=0).fit(df[["norm_bw", "norm_bh"]])
  unique, counts = np.unique(kmeans.labels_, return_counts=True)
  print(f"---- KMeans with K = {k} -----")
  counts = np.asarray((unique, counts)).T
  print(counts)

  print(f"---- Plot cluster distribution -----")
  count_df = pd.DataFrame(counts, columns=["cluster", "count"])

  plot_class_bar(count_df, x="cluster", y="count")

  df["cluster_labels"] = kmeans.labels_

  print(f"---- Get anchor box Multipliers -----")
  ab_scaled = df.apply(get_scale_factor, axis=1).to_list()
  ab_mult_df = pd.DataFrame(ab_scaled, columns=["ab_m_w", "ab_m_h"])
  df = pd.concat([df.reset_index(drop=True), ab_mult_df], axis=1)

  print(f" ----plot Anchor boxes on sample image---- ")
  plot_anchorboxes(image, kmeans)

  print(f" ----Visualize Anchor Box Clusters---- ")
  df["cluster_labels"] = df["cluster_labels"].astype('O')
  visualize_cluster(df, x="norm_bw", y="norm_bh")

  return(df.head())
