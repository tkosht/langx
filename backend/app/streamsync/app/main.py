import streamsync as ss
import pandas as pd 
from sklearn.datasets import load_iris
import plotly.express as px


# This is a placeholder to get you started or refresh your memory.
# Delete it or adapt it as necessary.
# Documentation is available at https://streamsync.cloud

# Shows in the log when the app starts
print("Hello world!")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
fig = px.scatter(df,x='sepal length (cm)', y='sepal width (cm)',color='target')

ss.init_state({
    "my_app":{
        "title":"StreamSync SampleApp"
    },
    "my_plotly_fig":fig,
    "dataframe":df
})


# Its name starts with _, so this function won't be exposed
def _update_message(state):
    is_even = state["counter"] % 2 == 0
    message = ("+Even" if is_even else "-Odd")
    state["message"] = message

def decrement(state):
    state["counter"] -= 1
    _update_message(state)

def increment(state):
    state["counter"] += 1
    # Shows in the log when the event handler is run
    print(f"The counter has been incremented.")
    _update_message(state)
    
# Initialise the state

# "_my_private_element" won't be serialised or sent to the frontend,
# because it starts with an underscore

initial_state = ss.init_state({
    "my_app": {
        "title": "Experiment App"
    },
    "_my_private_element": 1337,
    "message": None,
    "counter": 7,
    "figure":fig,
    "dataframe":df

})

_update_message(initial_state)