import pandas as pd
from IPython.display import display
from ipywidgets import Layout
from bokeh.models import CategoricalColorMapper
from bokeh.io import output_notebook
from ISYS5007_utilities import GoogleMap

pd.set_option("display.max_rows", None, "display.max_columns", None)

def rice_on_chess_board():
	total = 0
	lastSquare = 1
	numSquares=0

	table = []

	n=1
	for x in range(1, 9):
		for y in range(1, 9):
			numSquares = numSquares + 1
			total = total + lastSquare
			table.append([n, total, lastSquare])
			lastSquare = lastSquare*2
			n = n+1

	df=pd.DataFrame(table, columns=['Square Number', 'total', 'last square'])
	df.set_index('Square Number', inplace=True)
	return df


# Adapted from https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import ipywidgets as widgets

style = {'description_width' : 'initial'}

betaSlider = widgets.FloatSlider(value=0.3,min=0,max=1.0,step=0.1,description='Social distancing', style={'description_width' : '100px'}, layout=Layout(width='500px'))
daysSlider = widgets.FloatSlider(value=14,min=7,max=40,step=2.0,description='Days Contagious', style={'description_width' : '100px'}, layout=Layout(width='500px'))

def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I 
        return dSdt, dIdt, dRdt

def updatePlot(beta, days):
    doPlot(1.0-beta, 1./days)
    
# beta is infection rate
# gamma is rate peple removed from infectred pool
def doPlot(beta, gamma):
    # Total population
    N = 1000

    # Initial number of infected individuals
    I0 = 1

    # Initial number of removed individuals
    R0 = 0

    # S0 is everyone else who is initially suspeptible to infection
    S0 = N - I0 - R0

    # A grid of time points in days
    t = np.linspace (0, 160, 160)

    # Initial conditions vector
    y0 = S0, I0, R0

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    plt.plot(t, S, label = 'Uninfected')
    plt.plot(t, I, label = 'Infected')
    #plt.plot(t, R, label = 'Removed')
    plt.xlabel('Time days')
    plt.ylabel('Number')
    plt.legend()
    plt.show()

def SIR_model_UI():   
    widgets.interact(updatePlot, beta=betaSlider, days=daysSlider)

interface = [

{'title'      : '1.5 Meters of Distance',
 'image'      :  'images/distance.png',
 'copyright'  : 'www.pixnio.com',
 'description': 'Amy is an avid hiker. Hiking not only provides exercise, but getting close to nature helps her manage stress and keeps things in perspective. Social distancing is impossible on a narrow trail when passing other hikers. Amy will commit to walking in her neighborhood instead where she can stay 1.5 meters from others on the path.',
 'resource'   : 'https://www.health.gov.au/news/health-alerts/novel-coronavirus-2019-ncov-health-alert/how-to-protect-yourself-and-others-from-coronavirus-covid-19/social-distancing-for-coronavirus-covid-19',
 'checkbox'   : widgets.Checkbox(value=False, description='I will commit to keeping a 1.5 meter distance from others to Flatten the Curve.', layout=widgets.Layout(width='800px'))
},


{'title'      : 'Wash Hands',
 'image'      : 'images/wash_hands.png',
 'copyright'  : 'www.pixnio',
 'description': 'Kalinda is a single mother of three children. They live in a rural community. The community has been quarantined and open only to community residents in order to reduce Sars CoV 2 transmission from outside. There are nurses and small medical clinics available in this area but doctors are available mostly using telemedicine options. A severe COVID-19 illness would require air transport to a hospital. Kalinda teaches her children how to properly wash their hands. ',
 'resource'   : 'https://www.cdc.gov/coronavirus/2019-ncov/prevent-getting-sick/diy-cloth-face-coverings.html',
 'checkbox'   : widgets.Checkbox(value=False, description='I will commit to frequent hand washing to Flatten the Curve.', layout=widgets.Layout(width='800px'))
},

{'title'      : 'Wear a Face Mask',
 'image'      : 'images/facemask.png',
 'copyright'  : '',
 'description': 'Joe is taking care of his elderly mother, Mae. Since Mae is at an age with increased risk for COVID-19 serious illness, it is important that he doesn’t expose her to the virus. While out doing essential errands like shopping for food, wearing a face mask will help reduce the risk of Sars CoV 2 exposure. Joe makes his own face mask to cover his nose and mouth while out running essential errands.',
 'resource'   : 'https://www.cdc.gov/coronavirus/2019-ncov/prevent-getting-sick/diy-cloth-face-coverings.html',
 'checkbox'   : widgets.Checkbox(value=False, description='I will commit to wearing a face mask while doing essential errands to Flatten the Curve.', layout=widgets.Layout(width='800px'))
},

{'title'      : 'Wash and Disinfect Surfaces',
 'image'      : 'images/disinfect.png',
 'copyright'  : 'www.pixnio.com',
 'description': 'Joe is taking care of his elderly mother, Mae. Since Mae is at an age with increased risk for COVID-19 serious illness, it is important that he doesn’t expose her to the virus. While out doing essential errands like shopping for food, wearing a face mask will help reduce the risk of Sars CoV 2 exposure. Joe makes his own face mask to cover his nose and mouth while out running essential errands.',
 'resource'   : 'https://www.cdc.gov/coronavirus/2019-ncov/prevent-getting-sick/disinfecting-your-home.html',
 'checkbox'   : widgets.Checkbox(value=False, description='I will commit to regular cleaning of frequently touched surfaes to Flatten the Curve.', layout=widgets.Layout(width='800px'))
},

{'title'      : 'Keep a Supply of Food and Medicine',
 'image'      : 'images/medicine.png',
 'copyright'  : 'www.publicdomainfiles.com',
 'description': 'Joe is taking care of his elderly mother, Mae. Since Mae is at an age with increased risk for COVID-19 serious illness, it is important that he doesn’t expose her to the virus. While out doing essential errands like shopping for food, wearing a face mask will help reduce the risk of Sars CoV 2 exposure. Joe makes his own face mask to cover his nose and mouth while out running essential errands.',
 'resource'   : 'https://www.cdc.gov/coronavirus/2019-ncov/daily-life-coping/essential-goods-services.html',
 'checkbox'   : widgets.Checkbox(value=False, description='I will commit to minimising essential errand frequency to Flatten the Curve.', layout=widgets.Layout(width='800px'))
},

{'title'      : 'Make a Plan',
 'image'      : 'images/plan.png',
 'copyright'  : 'www.commons.wikimedia.org',
 'description': 'Andrea is a young professional. She lives close to her parents who are retired, and to her cousins who are essential workers. During the quarantine, Andrea is able to remotely work from home. Andrea, her parents, and her cousins have a plan. Their plan is made in case one of the households become ill with the COVID-19 virus. This plan details how this family network will safely supply groceries, supplies and medication to the infected household as well as who will provide care to sick individuals.',
 'resource'   : 'https://www.cdc.gov/coronavirus/2019-ncov/daily-life-coping/get-your-household-ready-for-COVID-19.html',
 'checkbox'   : widgets.Checkbox(value=False, description='I will make a plan to Flatten the Curve.', layout=widgets.Layout(width='800px'))
},

{'title'      : 'Stay Home',
 'image'      : 'images/home.png',
 'copyright'  : 'www.pixnio.com',
 'description': 'Andrea is a young professional. She lives close to her parents who are retired, and to her cousins who are essential workers. During the quarantine, Andrea is able to remotely work from home. Andrea, her parents, and her cousins have a plan. Their plan is made in case one of the households become ill with the COVID-19 virus. This plan details how this family network will safely supply groceries, supplies and medication to the infected household as well as who will provide care to sick individuals.',
 'resource'   : 'https://www.hopkinsmedicine.org/health/conditions-and-diseases/coronavirus/coronavirus-social-distancing-and-self-quarantine',
 'checkbox'   : widgets.Checkbox(value=False, description='I will commit to staying home to Flatten the Curve.', layout=widgets.Layout(width='800px'))
},

{'title'      : 'Manage Stress',
 'image'      : 'images/stress.png',
 'copyright'  : 'www.commons.wikimedia.org',
 'description': 'Kenneth is a nonessential gig worker. Quarantine has impacted his income. Ken is stressed about the COVID-19 epidemic. To take care of his health and minimize anxiety, Ken takes breaks from the news, meditates regularly, eats a healthy diet and gets physical activity daily.',
 'resource'   : 'https://nutrition.org/making-health-and-nutrition-a-priority-during-the-coronavirus-covid-19-pandemic/',
 'checkbox'   : widgets.Checkbox(value=False, description='I will commit to stress management to Flatten the Curve.', layout=widgets.Layout(width='800px'))
},

{'title'      : 'Stay Away from Others When Sick',
 'image'      : 'images/sick.png',
 'copyright'  : 'www.commons.wikimedia.org',
 'description': 'Jared is sick. He has a cough, shortness of breath, and a fever. When he coughs he coughs into his elbow. He stays at home and away from other people, and he calls his doctor.',
 'resource'   : 'https://www.cdc.gov/coronavirus/2019-ncov/if-you-are-sick/steps-when-sick.html',
 'checkbox'   : widgets.Checkbox(value=False, description='If I should become ill, I will stay at home and contact my physician to Flatten the Curve.', layout=widgets.Layout(width='800px'))
},
]

nameLabel  = widgets.Label(value="Name (optional)", layout=widgets.Layout(width='150px'))
nameText   = widgets.Text()
name       = widgets.HBox([nameLabel, nameText])

locationLabel  = widgets.Label(value="Location (optional)", layout=widgets.Layout(width='150px'))
locationText   = widgets.Text()
location       = widgets.HBox([locationLabel, locationText])

checkboxes = [item['checkbox'] for item in interface]
strategyText = widgets.Textarea(value="", layout=widgets.Layout(width='1024px', height='200px'))
submitButton = widgets.Button(description='Submit')
communityText = widgets.HTML(value="", layout=widgets.Layout(width='1024px', height='500px'))

# Save the note by wroting it to the file for the given sprint
def noteSaveCallback(button):
    newContents = "<p><strong>" + nameText.value + " from " + locationText.value + " writes:</strong></p>"
    newContents += "<p>" + strategyText.value + "</p>"
    newContents += readNote()
    communityText.value = newContents
    filename = './data/strategies.txt'
    f=open(filename, 'w')
    f.write(newContents)
    f.close()

# Read the contents of the note
def readNote():
    filename = './data/strategies.txt'
    contents = 'should not see this'
    try:
        f = open(filename, 'r')
        contents = f.read()
        f.close()
    except:
        contents = 'Provide your personal strategy here.'
        f = open(filename, 'w+')
        f.write(contents)
        f.close()
    return contents

def constructHTML(item):
    html = ""
    html += "<table style='padding:15px;'><tr>"
    html += "<td>"
    html += "<img width='320px' src='" + item['image'] + "'/></br>"
    html += item['copyright']
    html += "</td>"
    html += "<td  style='padding: 15px;' valign='top'>"
    html += "<h3>" + item['title'] + "</h3>"
    html += "<p style='line-spacing:0; padding: 0; margin: 0; color:green;'>"  + item['description'] + "</p>"
    html += "<strong>Resource:</br></strong> " + "<a href='" + item['resource'] + "'>" + item['resource'] + "</a>"
    html += "</td>"
    html += "</tr></table>"
    return widgets.VBox([widgets.HTML(value=html), item['checkbox']])

def commitmentUI():
    submitButton.on_click(noteSaveCallback)
    return widgets.VBox([name, location, widgets.VBox(checkboxes), widgets.Label(value='My Personal Strategy for Flattenting the Curve:'), strategyText, submitButton])


def communityUI():
    communityText.value = readNote()
    return communityText

def getData():
    nsw = pd.read_csv('data/covid-19-cases-by-notification-date-and-postcode-local-health-district-and-local-government-area.csv')
    coordinates = pd.read_csv('data/postcodes.csv', header=None)
    # Format NSW data

    # Drop 124 columns that contain at least 1 missing values
    # Keep only selected columns
    # Covert postcode to integer (i.e. remove decimal place)
    # Rename columns
    nsw.dropna(inplace=True)
    nsw = nsw[['postcode', 'lhd_2010_name', 'lga_name19']]
    nsw['postcode'] = nsw['postcode'].astype(int)
    nsw.rename(columns = {'postcode':'Postcode'}, inplace = True)

    # Format latitude, longitude data by postcode

    # Rename columns before merge and remove duplicates
    coordinates.rename(columns = {0:'Postcode', 1:'Suburb', 2:'State', 3:'Latitude', 4:'Longitude'}, inplace = True)
    coordinates.drop_duplicates(['Postcode'], inplace=True)

    # Merge latitude, longitude data with NSW data
    nsw = pd.merge(nsw, coordinates, on=['Postcode'])

    # Add country column for NSW data and create Total with value of 1 for each record
    nsw['Total'] = 1

    # Drop columns no longer needed
    nsw.drop(['Postcode', 'lhd_2010_name', 'lga_name19'], axis = 1, inplace=True) 

    # Group by Suburb and sum counts
    nsw = nsw.groupby(['Suburb', 'State', 'Latitude', 'Longitude'])['Total'].count().reset_index()

    return nsw

output_notebook()

# Update data/keys.csv with your personal keys
keys=pd.read_csv('data/keys.csv', encoding='utf-8')
keys.set_index('INDEX', inplace=True)

GOOGLE_KEY   = keys.loc['GOOGLE_KEY',   'KEY']

def mapData():

    data = getData()

    # Change these things:
    map_attributes=dict(
        center_latitude = -33.8688,  
        center_longitude= 151.209,
        zoom_factor     = 10,
        map_type        = "roadmap",
        map_title       = "COVID19 Infections",
        personal_key    = GOOGLE_KEY,
        data            = dict(
        latitude  = data['Latitude'].tolist(),
        longitude = data['Longitude'].tolist(),
        state  = data['State'].tolist(),
        suburb = data['Suburb'].tolist(),
        infections = data['Total'].tolist(),
        sizes     = data['Total']
        ),
        tooltips        = [
        ("Location", "@suburb"),
        ("State", "@state"),
        ('Total Infections', "@infections"),
        ],
        marker_size    = 10,
        fill_color     = "red"
    )


    return GoogleMap(map_attributes)