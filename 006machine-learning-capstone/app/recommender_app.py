import streamlit as st
import pandas as pd
import time

from wordcloud import STOPWORDS, WordCloud
import backend as backend
import numpy as np
import matplotlib.pyplot as plt


from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# Basic webpage setup
st.set_page_config(
   page_title="Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)


# ------- Functions ------
# Load datasets
@st.cache_data
def load_ratings():
    return backend.load_ratings()


@st.cache_data
def load_course_sims():
    return backend.load_course_sims()


@st.cache_data
def load_courses():
    return backend.load_courses()


@st.cache_data
def load_bow():
    return backend.load_bow()

@st.cache_data
def load_genres():
    return backend.load_genre()

# Initialize the app by first loading datasets
def init__recommender_app():

    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()
        # get course_genre
        course_genres_df = load_genres

    # Select courses
    st.success('Datasets loaded successfully...')

    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses: ")
    st.table(results)
    return results


def train(model_name, params):

    if model_name == backend.models[0]:
        # Start training course similarity model
        with st.spinner('Training...'):
            time.sleep(0.5)
            backend.train(model_name, params)
        st.success('Done!')
    # TODO: Add other model training code here
    elif model_name == backend.models[1]:
        # Start training model using user profile and course genres
        with st.spinner('Training... with user profile and course genres'):
            backend.train(model_name, params)
        st.success('Done!')
    else:
        pass


def predict(model_name, user_ids, params):
    res = None
    # Start making predictions based on model name, test user ids, and parameters
    with st.spinner('Generating course recommendations: '):
        time.sleep(0.5)
        res = backend.predict(model_name, user_ids, params)
    st.success('Recommendations generated!')
    return res


# ------ UI ------
# Sidebar
st.sidebar.title('Personalized Learning Recommender')
# Initialize the app
selected_courses_df = init__recommender_app()

# Model selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')
# Course similarity model
if model_selection == backend.models[0]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    # Add a slide bar for choosing similarity threshold
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold
# TODO: Add hyper-parameters for other models

# User profile model
elif model_selection == backend.models[1]:
    profile_sim_threshold = st.sidebar.slider('User Profile Similarity Threshold %',
                                              min_value=0, max_value=100,
                                              value=50, step=10)
    params['sim_threshold'] = profile_sim_threshold
# Clustering model
elif model_selection == backend.models[2]:
    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=0, max_value=50,
                                   value=20, step=1)
else:
    pass


# Training
st.sidebar.subheader('3. Training: ')
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.text('')
# Start training process
if training_button:
    train(model_selection, params)


# Prediction
st.sidebar.subheader('4. Prediction')
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button and selected_courses_df.shape[0] > 0:
    # Create a new id for current user session
    new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    user_ids = [new_id]
    res_df = predict(model_selection, user_ids, params)
    res_df = res_df[['COURSE_ID', 'SCORE']]
    course_df = load_courses()
    res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
    st.table(res_df)


# Course count per genres
st.sidebar.subheader("5. Course per genres")
course_per_genre_button = st.sidebar.button("Courses per Genre")
if course_per_genre_button:
    course_genres_df = load_genres()
    rs = course_genres_df.drop(["COURSE_ID", "TITLE"], axis=1).sum().sort_values()
    fig, ax = plt.subplots()
    rs.plot.bar()
    st.pyplot(fig)

# Course enrollment distribution​
st.sidebar.subheader("6. Course enrollment distribution​")
course_enrollment_distribution_button = st.sidebar.button("Course enrollment distribution​")
if course_enrollment_distribution_button:
    rating_df = load_ratings()
    ratingcount = rating_df.groupby(["user"]).size()
    fig, ax = plt.subplots()
    ratingcount.plot.hist(bins=100)
    st.pyplot(fig)


# Top 20 courses
st.sidebar.subheader("7. Top 20 courses")
top_20courses_button = st.sidebar.button("Top 20 courses")
if top_20courses_button:
    ratings_df = load_ratings()
    course_df = load_courses()
    newdf = pd.merge(ratings_df, course_df[['COURSE_ID', 'TITLE']], how='left', left_on='item',right_on='COURSE_ID')
    top20 = newdf[['TITLE','rating']].groupby('TITLE').count().sort_values(by='rating',ascending=False)[:20]
    st.table(top20)

# Word cloud of course titles​
st.sidebar.subheader("7. Word cloud of course titles​")
word_cloud_course_title_button = st.sidebar.button("Word cloud of course titles​")
if word_cloud_course_title_button:
    ratings_df = load_ratings()
    course_df = load_courses()
    titles = " ".join(title for title in course_df['TITLE'].astype(str))

    stopwords = set(STOPWORDS)
    stopwords.update(["getting started", "using", "enabling", "template", "university", "end", "introduction", "basic"])

    wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=800, height=400)
    wordcloud.generate(titles)
    st.image(wordcloud.to_image())

