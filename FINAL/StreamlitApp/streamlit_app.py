import streamlit as st

home_page = st.Page("home.py", title="Home", icon=":material/home:")
JobMatchingSystem_page = st.Page("JobMatchingSystem.py", title="Job Matching and Candidate Analysis System", icon=":material/assignment_turned_in:")

pg = st.navigation([home_page, JobMatchingSystem_page])
pg.run()