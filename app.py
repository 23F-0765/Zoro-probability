import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score




#configuring the streamlit page layout and dark theme, because we wanted the web interface to be dark
st.set_page_config(page_title="Game Sales Statistics", layout="wide", initial_sidebar_state="expanded")

#this is the code for a custom css and matplotlib dark theme
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #00ffcc;}
    .stDataFrame {border-radius: 10px;}
    div[data-testid="metric-container"] {
        background-color: #1e2530;
        border: 1px solid #00ffcc;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

#setting matplotlib to a dark theme so it matches the streamlit layout
plt.style.use('dark_background')




#this is where we loaded the data and did data cleaning, for example changing the user score to
#numeric data so mathematical calculations can be performed , also dropping null entries
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('Video_Games.csv')
    if 'User_Score' in df.columns:
        df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
    df = df.dropna(subset=['Global_Sales', 'Critic_Score', 'User_Score', 'Genre'])
    return df
df = load_and_clean_data()




#code for the sidebar for navigation
st.sidebar.title("Statistical Project")
st.sidebar.markdown("### Navigation Menu")
section = st.sidebar.radio("Go to:", [
    "0. Project Overview", # NEW
    "1. Tabular Representation",
    "2. Descriptive Statistics",
    "3. Exploratory Data Analysis (EDA)",
    "4. Probability & Distributions",
    "5. Hypothesis Testing & C.I.",
    "6. Linear Regression (OLS & ML)",
    "7. Classification Model",
    "8. Project Conclusion" # NEW
])
st.title("Video Game Sales & Ratings Analysis")






#Section 0: Project Overview
if section == "0. Project Overview":
    st.header("Welcome to the Meta-Matrix Data Project")
    st.image(
        "https://images.unsplash.com/photo-1542751371-adc38448a05e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80",
        use_column_width=True)

    st.markdown("""
    ### Project Objective
    This interactive application is designed to perform a comprehensive statistical analysis of the **Video Game Sales and Ratings** dataset. 

    Our goal is to apply rigorous mathematical modeling to understand what makes a video game a commercial success. We will explore:
    * **Descriptive Statistics:** Understanding central tendencies and dispersions in game scores.
    * **Probability:** Modeling how critic scores are distributed across the industry.
    * **Inferential Statistics:** Testing hypotheses regarding game genres and their market performance.
    * **Machine Learning:** Utilizing Ordinary Least Squares (OLS) Regression and Logistic Regression to predict global sales based on historical review data.

    Use the sidebar on the left to navigate through the different phases of our analysis.
    """)





# Section 1: Tabular Representation
if section == "1. Tabular Representation":
    st.header("1. Tabular Data Representation")
    genres = df['Genre'].unique()
    selected_genres = st.multiselect("Filter by Genre:", options=genres, default=genres[:3])

    if not selected_genres:
        st.warning("Please select at least one genre to display the data!")
    else:
        filtered_df = df[df['Genre'].isin(selected_genres)]

        # Interactive Dashboard Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Games Analyzed", len(filtered_df))
        col2.metric("Avg Critic Score", f"{filtered_df['Critic_Score'].mean():.1f}")
        col3.metric("Total Global Sales", f"{filtered_df['Global_Sales'].sum():.1f}M")

        st.dataframe(
            filtered_df[['Name', 'Platform', 'Year_of_Release', 'Genre', 'Global_Sales', 'Critic_Score', 'User_Score']],
            use_container_width=True)

# Section 2: Descriptive Statistics
elif section == "2. Descriptive Statistics":
    st.header("2. Measures of Central Tendency & Dispersion")
    num_cols = ['Global_Sales', 'Critic_Score', 'User_Score']
    selected_var = st.radio("Select a Variable to Analyze:", num_cols, horizontal=True)

    if selected_var:
        st.markdown("### Interactive Statistical Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean", f"{df[selected_var].mean():.2f}")
        col2.metric("Median", f"{df[selected_var].median():.2f}")
        col3.metric("Mode", f"{df[selected_var].mode()[0]:.2f}")
        col4.metric("Std Dev", f"{df[selected_var].std():.2f}")

        st.info(f"The variance for {selected_var} is calculated at **{df[selected_var].var():.2f}**.")





# Section 3: EDA
elif section == "3. Exploratory Data Analysis (EDA)":
    st.header("3. Graphical Representation (EDA)")
    st.write("Select a visualization from the menu below to explore different relationships in the data.")

    #interactive selection for plot types using a horizontal radio button
    plot_type = st.radio("Choose a Plot Type:", [
        "Histogram: Critic Score Distribution",
        "Bar Chart: Top Genres by Sales",
        "Scatter Plot: Score vs. Sales Correlation",
        "Box Plot: Sales Outliers by Platform",
        "Line Chart: Global Sales Over Time",
        "Pie Chart: Top Platforms Market Share",  # NEW
        "Heatmap: Correlation & Covariance"  # NEW
    ], horizontal=True)

    st.markdown("---")

    #1.Histogram
    if plot_type == "Histogram: Critic Score Distribution":
        st.subheader("Critic Score Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['Critic_Score'], bins=30, color='#00ffcc', edgecolor='black')
        ax.set_xlabel('Critic Score')
        ax.set_ylabel('Frequency')
        ax.set_title('How are Critic Scores Distributed?')
        st.pyplot(fig)
        st.info(
            "**Statistical Observation:** A histogram shows the shape of the data. Notices if the scores form a symmetric normal curve or if they are skewed to the left/right."
        )

    #2.Bar chart
    elif plot_type == "Bar Chart: Top Genres by Sales":
        st.subheader("Top 5 Genres by Global Sales")
        sales_by_genre = df.groupby('Genre')['Global_Sales'].sum().nlargest(5)
        fig, ax = plt.subplots(figsize=(8, 5))
        sales_by_genre.plot(kind='bar', color='#ff4b4b', ax=ax)
        ax.set_ylabel('Total Global Sales (Millions)')
        ax.set_title('Which Genres Sell the Most?')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.info(
            "**Statistical Observation:** Bar charts are excellent for comparing total volumes across qualitative categories."
        )

    #3.Scatter plot
    elif plot_type == "Scatter Plot: Score vs. Sales Correlation":
        st.subheader("Correlation: Critic Score vs. Global Sales")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df['Critic_Score'], df['Global_Sales'], alpha=0.5, color='#a020f0')
        ax.set_xlabel('Critic Score')
        ax.set_ylabel('Global Sales (Millions)')
        ax.set_title('Do higher scores mean more sales?')
        st.pyplot(fig)
        st.info(
            "**Statistical Observation:** Scatter plots help identify linear or non-linear correlation between two continuous variables."
        )

    #4.Box Plot
    elif plot_type == "Box Plot: Sales Outliers by Platform":
        st.subheader("Dispersion and Outliers: Sales by Top Platforms")
        #gets the top 5 platforms with the most games
        top_platforms = df['Platform'].value_counts().nlargest(5).index
        df_top_platforms = df[df['Platform'].isin(top_platforms)]

        fig, ax = plt.subplots(figsize=(8, 5))
        data_to_plot = [df_top_platforms[df_top_platforms['Platform'] == p]['Global_Sales'] for p in top_platforms]

        bp = ax.boxplot(data_to_plot, patch_artist=True, labels=top_platforms)
        for box in bp['boxes']:
            box.set(facecolor='#00ffcc', linewidth=2, alpha=0.7)
        for median in bp['medians']:
            median.set(color='#ff4b4b', linewidth=2)
        for flier in bp['fliers']:
            flier.set(marker='o', color='#ff4b4b', alpha=0.4)  #fliers are the outliers

        ax.set_ylabel('Global Sales (Millions)')
        ax.set_title('Sales Variance and Outliers per Platform')
        ax.set_ylim(0, 8)  #capping th y-axis so the boxes are readable, ignoring extreme outliers
        st.pyplot(fig)
        st.info(
            "**Statistical Observation:** Box plots show the Interquartile Range (IQR). The dots outside the main boxes represent 'outliers'—games that sold abnormally well compared to the median."
        )

    # 5.Line chart
    elif plot_type == "Line Chart: Global Sales Over Time":
        st.subheader("Time Series: Global Sales Over the Years")
        #group sales by year
        sales_by_year = df.groupby('Year_of_Release')['Global_Sales'].sum().reset_index()
        #cleans out anomalous future years if any exist in the dataset
        sales_by_year = sales_by_year[sales_by_year['Year_of_Release'] <= 2020]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sales_by_year['Year_of_Release'], sales_by_year['Global_Sales'], marker='o', color='#ff4b4b',
                linewidth=2)
        ax.set_xlabel('Year of Release')
        ax.set_ylabel('Total Global Sales (Millions)')
        ax.set_title('Gaming Industry Sales Trends')
        ax.grid(color='#1e2530', linestyle='--', linewidth=1)  # Adding a subtle grid
        st.pyplot(fig)
        st.info(
            "**Statistical Observation:** Line charts are the most effective method to visualize continuous chronological data to spot trends and life cycles."
        )

        #6.Pie chart
    elif plot_type == "Pie Chart: Top Platforms Market Share":
        st.subheader("Market Share of Top 5 Platforms")
        top_platforms = df['Platform'].value_counts().nlargest(5)

        fig, ax = plt.subplots(figsize=(6, 6))
        colors = ['#ff4b4b', '#00ffcc', '#a020f0', '#ffcc00', '#0099ff']
        ax.pie(top_platforms, labels=top_platforms.index, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'color': "w"})
        ax.set_title('Top 5 Platforms by Total Games Released')
        st.pyplot(fig)
        st.info(
            "**Statistical Observation:** Pie charts visualize parts-of-a-whole relationships, showing the dominance of legacy consoles in the historical dataset."
        )

    #7.Correlation Heatmap
    elif plot_type == "Heatmap: Correlation & Covariance":
        st.subheader("Measuring Association: Correlation Matrix")
        # Calculate correlation for continuous numerical variables
        corr_data = df[['Global_Sales', 'Critic_Score', 'User_Score']].corr()

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr_data, annot=True, cmap='mako', linewidths=.5, ax=ax, vmin=-1, vmax=1)
        ax.set_title("Pearson Correlation Coefficients")
        st.pyplot(fig)
        st.info(
            "**Statistical Observation:** A correlation closer to 1 indicates a strong positive relationship. We can see how strongly Critic Scores correlate with Global Sales.")





#Section 4: Probability and Distribution
elif section == "4. Probability & Distributions":
    st.header("4. Normal Distribution Application")
    st.write(
        "Let's analyze the **Critic Scores** to see if they follow a Normal Distribution, and calculate probabilities based on the area under the curve."
    )
    mean_score = df['Critic_Score'].mean()
    std_score = df['Critic_Score'].std()
    st.write(f"- **Population Mean (μ):** {mean_score:.2f}")
    st.write(f"- **Standard Deviation (σ):** {std_score:.2f}")

    #this is an interactive probability calculator, with a moveable bar to adjust the value accordingly
    st.subheader("Inverse Use of Standard Normal (Z-Score)")
    target_score = st.slider("Select a Critic Score to find its Probability:", 50, 100, 80)

    #calculating probability
    z_score = (target_score - mean_score) / std_score
    prob_greater = 1 - stats.norm.cdf(z_score)
    st.success(f"The Z-Score for a rating of {target_score} is **{z_score:.2f}**.")
    st.info(
        f"The probability of a random game getting a score HIGHER than {target_score} is **{prob_greater * 100:.2f}%**.")

    #normal curve plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.linspace(df['Critic_Score'].min(), df['Critic_Score'].max(), 100)
    p = stats.norm.pdf(x, mean_score, std_score)
    ax.plot(x, p, 'k', linewidth=2, color='#00ffcc')
    ax.fill_between(x, p, where=(x > target_score), color='#ff4b4b', alpha=0.5)
    ax.set_title('Normal Distribution of Critic Scores')
    st.pyplot(fig)






# Section 5: Hypothesis Testing and Confidence Interval
elif section == "5. Hypothesis Testing & C.I.":
    st.header("5. Confidence Intervals & Hypothesis Testing")
    st.subheader("A. Confidence Interval for Global Sales")
    confidence_level = st.radio("Select Confidence Level:", [0.90, 0.95, 0.99], index=1, horizontal=True)

    #calculating confidence interval
    sales_data = df['Global_Sales']
    mean_sales = np.mean(sales_data)
    sem = stats.sem(sales_data)  #standard error of mean
    margin_of_error = sem * stats.t.ppf((1 + confidence_level) / 2., len(sales_data) - 1)
    st.write(f"We are **{confidence_level * 100}% confident** that the true mean of Global Sales lies between:")
    st.success(f"**{mean_sales - margin_of_error:.2f} Million** and **{mean_sales + margin_of_error:.2f} Million**")
    st.markdown("---")
    st.subheader("B. Two-Sample Hypothesis Testing (Mean Testing)")
    st.write(
        "**Null Hypothesis (H0):** There is NO difference in average sales between Action games and Shooter games.")
    st.write("**Alternate Hypothesis (H1):** There IS a significant difference in average sales.")
    action_sales = df[df['Genre'] == 'Action']['Global_Sales']
    shooter_sales = df[df['Genre'] == 'Shooter']['Global_Sales']


    #T test independent
    t_stat, p_value = stats.ttest_ind(action_sales, shooter_sales, equal_var=False)
    st.write(f"- **T-Statistic:** {t_stat:.4f}")
    st.write(f"- **P-Value:** {p_value:.4e}")
    if p_value < 0.05:
        st.error(
            "Conclusion: Because the P-Value is less than 0.05, we **reject the Null Hypothesis**. The data proves a statistically significant difference in sales between Action and Shooter games.")
    else:
        st.warning("Conclusion: Because the P-Value is greater than 0.05, we **fail to reject the Null Hypothesis**.")

    #Hypothesis Distribution Graph
    st.markdown("#### Visualizing the Hypothesis")
    fig, ax = plt.subplots(figsize=(8, 4))

    #Plot overlapping distributions, capped at 5 million for readability
    ax.hist(action_sales, bins=30, range=(0, 5), alpha=0.5, color='#ff4b4b', label='Action Sales')
    ax.hist(shooter_sales, bins=30, range=(0, 5), alpha=0.5, color='#00ffcc', label='Shooter Sales')
    ax.set_xlabel('Global Sales (Millions)')
    ax.set_ylabel('Frequency')
    ax.set_title('Sales Distribution Comparison: Action vs. Shooter')
    ax.legend()
    st.pyplot(fig)




# Section 6: Linear Regression with OLS and Sklearn
elif section == "6. Linear Regression (OLS & ML)":
    st.header("6. Multiple Linear Regression")
    st.write("Predicting **Global Sales** based on **Critic Score** and **User Score**.")
    X = df[['Critic_Score', 'User_Score']]
    y = df['Global_Sales']
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Statistical OLS Model")

        #Adding a constant for the intercept
        X_sm = sm.add_constant(X)
        ols_model = sm.OLS(y, X_sm).fit()

        #Extracting the coefficients table (tables[1]) and converting it to a pandas dataframe
        results_table = ols_model.summary().tables[1]
        results_df = pd.DataFrame(results_table.data[1:], columns=results_table.data[0])

        #cleaning up the dataframe index for better presentation
        results_df.set_index(results_df.columns[0], inplace=True)
        results_df.index.name = "Variable"

        #displaying as a clean, responsive streamlit table
        st.dataframe(results_df, use_container_width=True)
        st.write(f"**R-squared:** {ols_model.rsquared:.4f}")
        st.caption("The P>|t| column shows if our variables are statistically significant (< 0.05).")

    with col2:
        st.subheader("Machine Learning Prediction (sklearn)")

        #Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #training sklearn model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        st.write("Test our model dynamically! Enter theoretical scores to predict global sales.")
        input_critic = st.slider("Select Critic Score:", 10, 100, 85)
        input_user = st.slider("Select User Score:", 1.0, 10.0, 8.0, step=0.1)

        #ensuring that our prediction doesn't go below zero
        prediction = lr_model.predict([[input_critic, input_user]])
        final_pred = max(0, prediction[0])
        st.success(f"Predicted Global Sales: **{final_pred:.2f} Million Copies**")

        # NEW: Regression Line/Scatter Visualization
        fig_reg, ax_reg = plt.subplots(figsize=(6, 4))
        ax_reg.scatter(df['Critic_Score'], df['Global_Sales'], color='gray', alpha=0.2, label='Dataset')
        ax_reg.scatter(input_critic, final_pred, color='#00ffcc', s=150, edgecolors='white', label='Your Prediction')

        # Plotting a rough trend line
        z = np.polyfit(df['Critic_Score'], df['Global_Sales'], 1)
        p = np.poly1d(z)
        ax_reg.plot(df['Critic_Score'], p(df['Critic_Score']), color='#ff4b4b', linestyle='--', label='Trend Line')
        ax_reg.set_xlabel('Critic Score')
        ax_reg.set_ylabel('Global Sales (Millions)')
        ax_reg.legend()
        st.pyplot(fig_reg)






#Section 7: Classification
elif section == "7. Classification Model":
    st.header("7. Machine Learning: Classification")
    st.write(
        "Using **Logistic Regression** to classify if a game will be a **Commercial Hit** (Over 1 Million Global Sales)."
    )

    #creating a binary target variable
    df['Is_Hit'] = (df['Global_Sales'] >= 1.0).astype(int)
    X = df[['Critic_Score', 'User_Score']]
    y = df['Is_Hit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #training logistic regression
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy:** {acc * 100:.2f}%")
    st.markdown("---")
    st.subheader("Interactive Hit Predictor")
    test_critic = st.number_input("Enter Expected Critic Score (0-100):", min_value=0, max_value=100, value=90)
    test_user = st.number_input("Enter Expected User Score (0-10):", min_value=0.0, max_value=10.0, value=8.5, step=0.1)

    if st.button("Predict Commercial Success"):
        hit_prediction = clf.predict([[test_critic, test_user]])
        hit_probability = clf.predict_proba([[test_critic, test_user]])[0][1]
        if hit_prediction[0] == 1:
            st.success(f"**Prediction: COMMERCIAL HIT!** (Probability: {hit_probability * 100:.1f}%)")
        else:
            st.error(f"**Prediction: NOT A HIT.** (Probability: {hit_probability * 100:.1f}%)")
            st.caption("Less than 1 million copies expected.")

#Section 8: Project Conclusion
elif section == "8. Project Conclusion":
    st.header("8. Conclusion & Key Findings")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.success("### The Impact of Critics")
        st.write(
            "Our regression models and correlation matrices confirm a statistically significant relationship between high **Critic Scores** and higher **Global Sales**. While not a perfect 1:1 predictor, critical reception heavily dictates market performance.")

    with col2:
        st.info("### Genre Dominance")
        st.write(
            "Through hypothesis testing, we proved that genre deeply impacts revenue. The variance in sales between genres like **Action** and **Shooter** is not due to random chance, but represents distinct consumer market sizes.")

    st.markdown("""
    ### Final Thoughts
    By combining **Exploratory Data Analysis (EDA)** with **Machine Learning (OLS and Logistic Regression)**, we transformed raw, unorganized gaming data into actionable intelligence. This project demonstrates the profound importance of data-driven decision-making in the modern entertainment industry.
    """)
