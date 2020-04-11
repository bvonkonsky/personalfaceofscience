import json
import sys
import re
import io
import csv
import pandas as pd
import datetime
import random
import numpy as np
import math

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from bs4 import BeautifulSoup
import requests

# Before insgtalling additional packages, it is recommended that you update Anaconda
# conda update anaconda
#
# The following pacakges must be added to the Anaconda python distribution
# conda install -c conda-forge tweepy
# conda install -c bokeh bokeh
# conda install -c conda-forge exifread

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream, API
import xml.etree.ElementTree as ET
import exifread
from bokeh.io import output_notebook, show
from bokeh.models import GMapPlot, GMapOptions, ColumnDataSource, Range1d, DataRange1d, Circle, PanTool, WheelZoomTool
from bokeh.models import TapTool, HoverTool, OpenURL

version='1.2 (25 March 2018 BvK)'
  
def scrape_attribute(html, tag, attributes, include_text=False):
  soup = BeautifulSoup(html, 'html.parser')
  tags = soup.find_all(tag)
  dictionary = [{attribute: tag.attrs[attribute] for attribute in attributes if tag.has_attr(attribute)} for tag in tags]
  return dictionary

def print_scraped_tables(table):
  for i in range(len(table)):
    print("table", i)
    for j in range(len(table[i])):
    	print('[' + str(j) + '] ', table[i][j]['row'])
    print()

def append_col(thelist, cols):
  if len(thelist) < cols:
    for i in range(len(thelist), cols):
      thelist.append('')
  return thelist

def get_colspan(tag):
  try:
    col = int(tag["colspan"])
  except (ValueError, KeyError) as e:
    col = 1
  return col

def make_dictionary(df, key, value):
  return df.set_index(key)[value].to_dict()

def aligncols(headerspan, rowspan, row):
  ptr = 0
  seen_cols = 0
  total_cols = sum(headerspan)
  new_row = []
  for i in range(len(headerspan)):
    if seen_cols >= total_cols:
      col = ''
    else:
      this_header  = headerspan[i]
      if ptr < len(rowspan):
        this_col     = rowspan[ptr]
        col =[ row[ptr] ]
        ptr  = ptr + 1
      while this_col < this_header and ptr<len(rowspan):
        this_col = this_col + rowspan[ptr]
        col.append(row[ptr])
        ptr = ptr + 1
      if len(col) == 1:
        col = col[0]
    new_row.append(col)
    seen_cols = seen_cols + this_col
  return new_row

def scrape_tag(html, tag, classname=None, return_text=True):
  soup = BeautifulSoup(html, 'html.parser')
  if classname is None:
    tags = soup.find_all(tag)
  else:
    tags = soup.find_all(tag, {'class': classname})
  if return_text == True:
    return [tag.text.strip() for tag in tags]
  else:
    return tags

def scrape_tables(html):
  # Parse the html and produce a BeutifulSoup object used in this function
  soup = BeautifulSoup(html, 'html.parser')

  # Replace all <br> tags with sapes
  for br in soup.find_all("br"):
    br.replace_with(" ")

  # For each table find all rows
  tables = []
  for table in soup.find_all('table'):
    data = []
    for tr in table.find_all('tr'):
      # In each row, find all header and data tags and determine the columns they span
      tags    = tr.find_all(['th', 'td'])
      row     = [tag.text.strip() for tag in tags]
      colspan = [get_colspan(tag) for tag in tags]

      # Append these to a list for the current table
      data.append({'row': row, 'colspan' : colspan})

    # Append data for each table to a list of all tables
    tables.append(data)
    
  return tables

def scraped_table_to_df(data, colspan_index = None, header_index=None, drop_index=None):
  # Pluck significant items out of the table if specified in parameter list
  if header_index is None:
    header = None
  else:
    header = data[header_index]['row']
  if colspan_index is None:
    colspan = None
  else:
    colspan = data[colspan_index]['colspan']

  # Get rid of any rows unwanted rows
  if drop_index is not None:
    if type(drop_index) is not list:
      drop_index = [ drop_index ]
    drop_index.sort(reverse=True)
    for pos in drop_index:
      del data[pos]

  # Align rows if necessary
  if colspan is not None:
    for pos in range(len(data)):
      data[pos]['row'] = aligncols(colspan, data[pos]['colspan'], data[pos]['row'])

  return pd.DataFrame([record['row'] for record in data], columns = header)


def repeat_to_length(string_to_expand, length):
  return (string_to_expand * (int(length/len(string_to_expand))+1))[:length]

def add_html(indentation, open_tag, value, close_tag, sep):
  indent = repeat_to_length('   ', indentation)
  line = indent + open_tag + str(value) + close_tag + sep
  return line

def df2html(df):
  html = []
  header = df.columns.tolist()
  html.append(add_html(0, '<html>', '', '', '\n'))
  html.append(add_html(1, '<body>', '', '','\n'))
  html.append(add_html(2, '<table>', '', '', '\n'))

  html.append(add_html(3, '<tr>', '', '', '\n'))
  for col in header:
    html.append(add_html(6, '<th>', col, '</th>', '\n'))
  html.append(add_html(3, '', '','</tr>', '\n'))
    
  for  index in df.index.values.tolist():
    row = df.ix[index].tolist()
    html.append(add_html(3, '<tr>', '', '', '\n'))
    for col in row:
        html.append(add_html(6, '<td>', col, '</td>', '\n'))
    html.append(add_html(3, '', '','</tr>', ''))

  html.append(add_html(2, "", '', '</table>', '\n'))
  html.append(add_html(1, "", '', '</body>','\n'))
  html.append(add_html(0, '', '', '</html>', "\n"))
  return html

def get_unique(df, column):
  categories = list(set(df[pd.notnull(df[column])][column]))
  if len(df[pd.isnull(df[column])][column]) > 0:
    categories.append('Unclassified')
  categories.sort()
  return categories

def count_unique(df, column):
  category_labels = get_unique(df, column)
  counted_categories = {label : len(df[df[column]==label]) for label in category_labels}
  unclassified_count = len(df[pd.isnull(df[column])][column])
  if unclassified_count > 0:
    counted_categories.update({'Unclassified' : unclassified_count})
  return pd.DataFrame(pd.Series(counted_categories), columns=['Total'])

def random_df(number, magnitude):
    random.seed()
    d = {'A': random.sample(range(number), magnitude), 'B': random.sample(range(number), magnitude)}
    return  pd.DataFrame(d)

def all_ints(col):
  int_found = True
  for data in col:
    try:
      cast = int(data)
    except:
      int_found = False
  return int_found

def cast_int(col):
  for i in range(0, len(col)):
    col[i]= int(col[i])
  return col

def all_floats(col):
  float_found = True
  for data in col:
    try:
      cast = float(data)
    except:
      float_found = False
  return float_found

def cast_float(col):
  for i in range(0, len(col)):
    col[i]= float(col[i])
  return col

def read_csv(filename):
  table = []
  header= []
  dictionary = {}
  with open(filename, "rU") as ifile:
    # Create a CSV reader object that implements the iterator protocol
    reader = csv.reader(ifile)
    rownum = 0

    # Consider each row in the file, which the csv reader returns as a list
    for row in reader:
      # If this is beginning of the file, remember the header\
      if rownum == 0:
        for col in range(0, len(row)):
          header.append(row[col])
          table.append([])
      else:
        # Print each cell of the row in the right column 
        for col in range(0, len(row)):
          table[col].append(row[col])
      rownum = rownum + 1

    for col in range(0, len(header)):
      if all_ints(table[col]):
        table[col] = cast_int(table[col])
      elif all_floats(table[col]):
        table[col] = cast_float(table[col])

    for col in range(0, len(header)):
      dictionary.update({header[col] : table[col]})

  return header, dictionary

def trim_string(data, width):
  ellipse_width = width - 4
  data = str(data)
  trimmed = (data[:ellipse_width] + '.. ') if len(data) >= width else data
  trimmed = trimmed.ljust(width)
  return trimmed

def print_table(header, table, colwidth=9):
  # Print the header
  row_string = ''
  for col in range(0, len(header)):
      row_string = row_string + trim_string(header[col], colwidth)
  print(row_string)

  # Print the rest of the table
  for row in range(0, len(table[header[0]])):
    row_string = ''
    for col in range(0, len(header)):
      feature = header[col]
      row_string = row_string + trim_string(table[feature][row], colwidth)
    print(row_string)

def check_one_key(data, key):
  msg = ''
  if  key in data:
    if data[key] is not None:
      msg = data[key]
  return msg

def check_two_keys(data, key1, key2):
  msg = ''
  if key1 in data:
    if data[key1] is not None:
      if key2 in data[key1]:
        if key2 is not None:
          msg = data[key1][key2]
  return msg

def check_three_keys(data, key1, key2, key3):
  msg = ''
  if key1 in data:
    if data[key1] is not None:
      if key2 in data[key1]:
        if key2 is not None:
          if key3 in data[key1][key2]:
            if key3 is not None:
              msg = data[key1][key2][key3]
  return msg

# Define class Listener, which is a kind of StreamListener
class Listener(StreamListener):

   def __init__(self, num_to_collect=10, filename='out.csv'):
      self.tweet_count = 0
      self.num_to_collect = num_to_collect
      self.filename = filename
      self.df = pd.DataFrame(columns=['created_at', 'name', 'screen_name', 'text'])

	#Override StreamListener on_data callback invoked when new data is ready
   def on_data(self, data):
      data = json.loads(data)
      created_at = check_one_key(data,  'created_at')
      name       = check_two_keys(data, 'user', 'name')
      screen_name= check_two_keys(data, 'user', 'screen_name')
      text       = check_three_keys(data, 'retweeted_status', 'extended_tweet', 'full_text')
      #coord      = check_two_keys(data, 'coordinates', 'coordinates')
      #if type(coord) is list:
      #  print('coord:', coord)

      self.df.loc[self.tweet_count]=[created_at, name, screen_name, text]

      # Increment the count and check if we've read all necessary
      self.tweet_count = self.tweet_count + 1
      if self.tweet_count < self.num_to_collect:
         return True
      else:
         print ('Finished collecting', self.num_to_collect, 'tweets on', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
         self.df.to_csv(self.filename, index=False, encoding='utf-8')
         return False

   def on_error(self, status):
        print ('Error on status', status)

   def on_limit(self, status):
        print ('Limit threshold exceeded', status)

   def on_timeout(self, status):
        print ('Stream timeout continuing...')


def start_listening(auth, filter=[], num_to_collect=2, filename='out.csv'):
   # Report when we start listening
   print ("Start listening on", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
   # Start listening
   stream = Stream(auth, Listener(num_to_collect=num_to_collect, filename=filename))
   # Set the filte
   stream.filter(track=filter)

def get_XML_root(xml):
  it = ET.iterparse(io.BytesIO(xml))
  for _, el in it:
    if '}' in el.tag:
      el.tag = el.tag.split('}', 1)[1]
  root = it.root
  return root

def xml2df(xml_data, data_tag='item'):
  #root = ET.fromstring(xml_data)
  root = get_XML_root(xml_data)

  all_rows=[]
  for item in root.iter(data_tag):
    row={}
    for record in item:
      tag = record.tag
      text = [ ]

      # Get the text if it exists and strip white space
      if record.text is not None:
        newtext = record.text
        newtext = newtext.strip()
        if len(newtext) > 0:
          text.append(newtext)

      # include hypertext link if one exists
      href = record.get('href')
      if href is not None:
        text.append(href)

      # Get any childrew that exist
      children = list(record)
      for child in list(record):
        if child.text is not None:
          text.append(child.text)

      # Covert back to text if only one entry in list   
      if len(text) == 1:
        text = text[0]

      if tag in row:
        if not isinstance(row[tag], list):
          newlist = []
          newlist.append(row[tag])
          row[tag]= newlist
        existing_list = row[tag]
        existing_list.append(text)
        row[tag] = existing_list
      else:
        row[record.tag]=text

    all_rows.append(row)
  return pd.DataFrame(all_rows)

def slope (x1, y1, x2, y2):
    rise = y2-y1
    run  = x2-x1
    return rise / run

def genPalette(minValue, maxValue, minColor, maxColor , steps): 
  r1, g1, b1 = minColor
  r2, g2, b2 = maxColor

  mR = slope (0, r1, steps, r2)
  mG = slope (0, g1, steps, g2)
  mB = slope (0, b1, steps, b2)
    
  palette = []
  for i in range (0, steps+1):
    index = i/steps  * (maxValue-minValue) + minValue
    rI = int((r1 + mR * i) * 255)
    gI = int((g1 + mG * i) * 255)
    bI = int((b1 + mB * i) * 255)
    palette.append ("#" "%0.2X"%rI + "%0.2X"%gI + "%0.2X"%bI)

  return palette

def convertToDecimal(raw):
  # component values store numerator (num) and denominator (den)
  degrees = float (raw[0].num) / raw[0].den
  minutes = float (raw[1].num) / raw[1].den
  seconds = float (raw[2].num) / raw[2].den
   
  decDeg  = degrees + minutes / 60.0 + seconds / 3600.0
  return decDeg

def getValue(dictionary, key):
  value = None
  if key in dictionary:
    value = dictionary[key].values
  return value

def getEXIF(img):
  f =open(img, 'rb')
  tags = exifread.process_file(f)
  f.close()
  return tags

def imageData(img):

  tags= getEXIF(img)
  
  latitude=None
  longitude=None
  GPSKeys = ['GPS GPSLatitude', 'GPS GPSLongitude', 'GPS GPSLatitudeRef', 'GPS GPSLongitudeRef']
  if all (key in tags for key in GPSKeys):
    latitude  = convertToDecimal(tags['GPS GPSLatitude'].values)
    longitude = convertToDecimal(tags['GPS GPSLongitude'].values)

    if tags['GPS GPSLatitudeRef'].printable  == 'S':
      latitude = -latitude

    if tags['GPS GPSLongitudeRef'].printable != 'E':
      longitude = -longitude
  

  theDate = getValue(tags, 'EXIF DateTimeOriginal')
  make    = getValue(tags, 'Image Make')
  model   = getValue(tags, 'Image Model')
  software= getValue(tags, 'Image Software')
  
  return [img, latitude, longitude, theDate, make, model, software]

def GoogleMap(ma):
  
  # Set attributes of the map, including the latitude and longitude, and zoom factor
  map_options = GMapOptions(lat=ma['center_latitude'], lng=ma['center_longitude'], map_type=ma['map_type'], zoom=ma['zoom_factor'])

  # Prepare a Google Map plot using default map ranges in both dimensions with the specified map options
  #plot = GMapPlot(x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options)
  plot = GMapPlot(x_range=Range1d(), y_range=Range1d(), map_options=map_options)
  plot.title.text = ma['map_title']

  # Replace the value below with your personal API key:
  # https://developers.google.com/maps/documentation/javascript/get-api-key
  plot.api_key = ma['personal_key']

  # Create a dictionary that has the data to be ploted over the map from DataFrame columns
  source = ColumnDataSource(ma['data'])

  # Generate and plot circle glyphs for each image location
  circle = Circle(x="longitude", y="latitude", size=ma['marker_size'], fill_color=ma['fill_color'], line_color=None)

  hover = HoverTool(tooltips=ma['tooltips'])

  plot.add_glyph(source, circle)
  plot.add_tools(PanTool(), WheelZoomTool(), hover)

  # Output to this Jupyter notebook
  #output_notebook()
  show(plot)

# The followoing is based on code from https://gist.github.com/yanofsky/5436496
def get_user_tweets(auth, screen_name, filename='usertweets.csv'):
	api = API(auth)
	
	#initialize a list to hold all the tweepy Tweets
	alltweets = []	
	
	#make initial request for most recent tweets (200 is the maximum allowed count)
	new_tweets = api.user_timeline(screen_name = screen_name,count=200)
	
	#save most recent tweets
	alltweets.extend(new_tweets)
	
	#save the id of the oldest tweet less one
	oldest = alltweets[-1].id - 1
	
	#keep grabbing tweets until there are no tweets left to grab
	while len(new_tweets) > 0:
		print ('getting tweets before id', oldest)
		
		#all subsiquent requests use the max_id param to prevent duplicates
		new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
		
		#save most recent tweets
		alltweets.extend(new_tweets)
		
		#update the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1
		
		print (len(alltweets), 'tweets downloaded so far')
	
	#transform the tweepy tweets into a 2D array that will populate the csv	
	#outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]

	outtweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]
	#write the csv	
	with open(filename, 'w', encoding='utf-8') as f:
		writer = csv.writer(f)
		writer.writerow(["id","created_at","text"])
		writer.writerows(outtweets)

def print_tweets(df):
  for index, row in df.iterrows():
    if 'id' in row.keys():
      print ("id:", row['id'])
    if 'created_at' in row.keys():
      print ('created_at:', row['created_at'])
    if 'text' in row.keys():
      print (row['text'])
    print ()


def linear_regression(X, y):
    regression  = LinearRegression().fit(X.values.reshape(-1,1), y)
    return regression.coef_[0], regression.intercept_

def plot_regression(data, title, xlabel, ylabel, figsize=(5, 4)):
    handles = []
    plt.figure(figsize=figsize)
    for entry in data:
        X=entry[0]
        y=entry[1]
        label=entry[2]
        color=entry[3]
        coefficient, intercept = linear_regression(X, y)
        plt.scatter(X, y, marker='o', s=50, color=color, alpha=0.5)
        plt.plot(X, coefficient * X + intercept, color=color)
        handles.append(mpatches.Patch(color=color, label=label))
    if len(handles) > 1:
      plt.legend(handles=handles)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def predict(X, coefficient, intercept):
    return X*coefficient + intercept
