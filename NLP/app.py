from pathlib import Path

import pandas as pd
import streamlit as st

from skill_extractor import extract_skills


# Page configuration

st.set_page_config(
    page_title="Job Skill Extractor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)



# Styling

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 18px;
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2rem;
        }

        .hero p {
            margin: 0;
            opacity: 0.78;
            font-size: 1rem;
        }

        .skill-chip {
            display: inline-block;
            padding: 0.38rem 0.72rem;
            margin: 0.22rem 0.20rem 0.22rem 0;
            border-radius: 999px;
            border: 1px solid rgba(128, 128, 128, 0.28);
            font-weight: 600;
            font-size: 0.92rem;
        }

        .small-note {
            opacity: 0.72;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Data loading

APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "cleaned_job_dataset.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def build_skill_frequency(descriptions: tuple[str, ...]) -> pd.Series:
    all_skills = []

    for description in descriptions:
        all_skills.extend(extract_skills(description))

    if not all_skills:
        return pd.Series(dtype="int64")

    return pd.Series(all_skills).value_counts()


@st.cache_data
def build_job_skill_table(
    titles: tuple[str, ...],
    descriptions: tuple[str, ...],
) -> pd.DataFrame:
    rows = []

    for title, description in zip(titles, descriptions):
        skills = extract_skills(description)
        for skill in skills:
            rows.append({"job_title": title, "skill": skill})

    if not rows:
        return pd.DataFrame(columns=["job_title", "skill"])

    return pd.DataFrame(rows)


try:
    df = load_data(str(DATA_FILE))
except FileNotFoundError:
    st.error(
        "Dataset file not found. Keep `cleaned_job_dataset.csv` in the same "
        "folder as `job_skill_web_app.py`."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

# LOGIN PAGE


OWNER_NAME = "Shraddha Santiur"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def login():
    st.markdown(
        """
        <div style="
            text-align:center;
            padding:20px;
        ">
            <h1>💼 JobSkill Insight</h1>
            <h2> Welcome To Our Page!</h2>
            <p>Login to continue</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button(
            "Login",
            type="primary",
            use_container_width=True
        ):

         if (
             username.strip() != "" 
         and password.strip() != ""
         ):

          st.session_state.logged_in = True
          st.session_state.username = username

         st.success(f"Welcome {username}!")
         st.rerun()

        else:
         st.error(
            "Please enter username and password."
            )   

        st.markdown("---")

        st.markdown(
            f"""
            <div style="text-align:center;">
                <b>App Owner</b><br>
                {OWNER_NAME}
            </div>
            """,
            unsafe_allow_html=True
        )


if not st.session_state.logged_in:

    login()

    st.stop()

# Sidebar

st.sidebar.title("💼 Job Skill App")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Extract Skills",
        "Dataset Explorer",
        "Skill Analysis",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "Built with Streamlit, Pandas and a dictionary-based skill extractor."
)



# Header

st.markdown(
    """
    <div class="hero">
        <h1>💼 Job Skill Extractor & Analytics</h1>
        <p>
            Extract technical skills from job descriptions and explore skill
            demand across the job-posting dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)



# Dashboard

if page == "Dashboard":
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Job Postings", f"{len(df):,}")
    c2.metric(
        "Job Titles",
        f"{df['job_title'].nunique():,}" if "job_title" in df.columns else "—",
    )
    c3.metric(
        "Companies",
        f"{df['company'].nunique():,}" if "company" in df.columns else "—",
    )
    c4.metric(
        "Locations",
        f"{df['location'].nunique():,}" if "location" in df.columns else "—",
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Top Job Titles")

        if "job_title" in df.columns:
            top_titles = df["job_title"].value_counts().head(10)
            st.bar_chart(top_titles)
        else:
            st.info("The dataset does not contain a `job_title` column.")

    with right:
        st.subheader("Job Type Distribution")

        if "job_type" in df.columns:
            job_types = df["job_type"].value_counts()
            st.bar_chart(job_types)
        else:
            st.info("The dataset does not contain a `job_type` column.")

    st.subheader("Sample Job Postings")
    preferred_columns = [
        "job_id",
        "job_title",
        "company",
        "location",
        "experience",
        "education",
        "job_type",
    ]
    available_columns = [c for c in preferred_columns if c in df.columns]

    st.dataframe(
        df[available_columns].head(15),
        use_container_width=True,
        hide_index=True,
    )



# Extract skills

elif page == "Extract Skills":
    st.subheader("Extract Skills From a Job Description")

    example_text = (
        "We are looking for a Data Analyst with experience in Python, SQL, "
        "Excel, Power BI, Pandas and data visualization. Knowledge of AWS "
        "and Git is a plus."
    )

    job_description = st.text_area(
        "Paste a job description:",
        height=250,
        placeholder=example_text,
    )

    col1, col2 = st.columns([1, 4])

    with col1:
        extract_clicked = st.button(
            "🔍 Extract Skills",
            type="primary",
            use_container_width=True,
        )

    with col2:
        use_example = st.button("Use Example Description")

    if use_example:
        job_description = example_text
        st.session_state["example_description"] = example_text
        st.info(
            "Example loaded. Copy it into the box above, or paste your own "
            "job description."
        )

    if extract_clicked:
        if not job_description.strip():
            st.warning("Please enter a job description.")
        else:
            skills = extract_skills(job_description)

            if skills:
                st.success(f"{len(skills)} matching skill(s) found.")

                chips = " ".join(
                    f'<span class="skill-chip">✅ {skill}</span>'
                    for skill in skills
                )
                st.markdown(chips, unsafe_allow_html=True)

                result_df = pd.DataFrame(
                    {
                        "Skill": skills,
                        "Matched": ["Yes"] * len(skills),
                    }
                )

                st.download_button(
                    "⬇️ Download Extracted Skills CSV",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="extracted_skills.csv",
                    mime="text/csv",
                )
            else:
                st.info(
                    "No matching skills were found in the current skill dictionary."
                )

    st.divider()

    st.subheader("Try a Job Description From the Dataset")

    if {"job_title", "job_description"}.issubset(df.columns):
        titles = sorted(df["job_title"].dropna().unique().tolist())
        selected_title = st.selectbox("Choose a job title", titles)

        matching = df[df["job_title"] == selected_title]

        if not matching.empty:
            row_number = st.number_input(
                "Posting number",
                min_value=1,
                max_value=len(matching),
                value=1,
                step=1,
            )

            selected_row = matching.iloc[int(row_number) - 1]
            description = str(selected_row["job_description"])

            st.text_area(
                "Dataset job description",
                value=description,
                height=220,
                disabled=True,
            )

            dataset_skills = extract_skills(description)

            if dataset_skills:
                chips = " ".join(
                    f'<span class="skill-chip">{skill}</span>'
                    for skill in dataset_skills
                )
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.caption("No dictionary skills were detected in this posting.")



# Dataset explorer

elif page == "Dataset Explorer":
    st.subheader("Explore Job Postings")

    filtered = df.copy()

    f1, f2, f3 = st.columns(3)

    with f1:
        if "job_title" in df.columns:
            title_options = ["All"] + sorted(
                df["job_title"].dropna().unique().tolist()
            )
            title_filter = st.selectbox("Job Title", title_options)

            if title_filter != "All":
                filtered = filtered[filtered["job_title"] == title_filter]

    with f2:
        if "job_type" in df.columns:
            type_options = ["All"] + sorted(
                df["job_type"].dropna().unique().tolist()
            )
            type_filter = st.selectbox("Job Type", type_options)

            if type_filter != "All":
                filtered = filtered[filtered["job_type"] == type_filter]

    with f3:
        if "location" in df.columns:
            location_options = ["All"] + sorted(
                df["location"].dropna().unique().tolist()
            )
            location_filter = st.selectbox("Location", location_options)

            if location_filter != "All":
                filtered = filtered[filtered["location"] == location_filter]

    search_text = st.text_input(
        "Search title, company, location or description",
        placeholder="Example: Python, Data Scientist, Remote...",
    )

    if search_text.strip():
        searchable_columns = [
            c
            for c in [
                "job_title",
                "company",
                "location",
                "job_description",
            ]
            if c in filtered.columns
        ]

        mask = pd.Series(False, index=filtered.index)

        for column in searchable_columns:
            mask = mask | filtered[column].astype(str).str.contains(
                search_text,
                case=False,
                na=False,
                regex=False,
            )

        filtered = filtered[mask]

    st.metric("Matching Postings", f"{len(filtered):,}")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    st.download_button(
        "⬇️ Download Filtered Dataset",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_job_dataset.csv",
        mime="text/csv",
    )


# ------------------------------------------------------------
# Skill analysis
# ------------------------------------------------------------
elif page == "Skill Analysis":
    st.subheader("Skill Demand Analysis")

    if "job_description" not in df.columns:
        st.error("The dataset does not contain a `job_description` column.")
        st.stop()

    descriptions = tuple(df["job_description"].fillna("").astype(str).tolist())
    skill_counts = build_skill_frequency(descriptions)

    if skill_counts.empty:
        st.info("No skills from the dictionary were detected in the dataset.")
        st.stop()

    max_skills = min(30, len(skill_counts))
    top_n = st.slider(
        "Number of skills to display",
        min_value=5,
        max_value=max_skills,
        value=min(20, max_skills),
    )

    top_skills = skill_counts.head(top_n)

    c1, c2, c3 = st.columns(3)
    c1.metric("Unique Skills Detected", f"{len(skill_counts):,}")
    c2.metric("Most Common Skill", str(skill_counts.index[0]))
    c3.metric("Mentions", f"{int(skill_counts.iloc[0]):,}")

    st.subheader(f"Top {top_n} Skills")
    st.bar_chart(top_skills)

    skill_table = (
        top_skills.rename("Number of Job Postings")
        .rename_axis("Skill")
        .reset_index()
    )

    st.dataframe(
        skill_table,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download Skill Frequency CSV",
        data=skill_table.to_csv(index=False).encode("utf-8"),
        file_name="skill_frequency.csv",
        mime="text/csv",
    )

    if {"job_title", "job_description"}.issubset(df.columns):
        st.divider()
        st.subheader("Top Skills by Job Title")

        title_options = sorted(df["job_title"].dropna().unique().tolist())
        selected_title = st.selectbox(
            "Select a job title for skill analysis",
            title_options,
            key="skill_analysis_title",
        )

        title_df = df[df["job_title"] == selected_title]
        title_descriptions = tuple(
            title_df["job_description"].fillna("").astype(str).tolist()
        )
        title_counts = build_skill_frequency(title_descriptions).head(15)

        if not title_counts.empty:
            st.bar_chart(title_counts)

            st.caption(
                f"Based on {len(title_df)} posting(s) with the title "
                f"“{selected_title}”."
            )
        else:
            st.info("No dictionary skills were detected for this job title.")



# Footer

st.divider()
st.markdown(
    '<p class="small-note">Job Skill Extractor • Streamlit Web Application</p>',
    unsafe_allow_html=True,
)
#streamlit run app.py
#We are looking for a Data Scientist proficient in Python, machine learning, deep learning, and SQL. Experience with Power BI is required.",
    #"Hiring a Senior Data Engineer with strong experience in Apache Spark, Python, SQL Server, and data warehousing techniques.",
    #"Seeking a Business Analyst skilled in SQL, Power BI, data analysis, and advanced Excel spreadsheet modeling."