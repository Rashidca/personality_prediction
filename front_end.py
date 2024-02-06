import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
import numpy as np
from nltk import TweetTokenizer
import joblib
from wordcloud import WordCloud
lematizer=WordNetLemmatizer()
nltk.download("punkt")
# Download the WordNet data
nltk.download('wordnet')
tk=TweetTokenizer()
nltk.download('stopwords')
sw=stopwords.words('english')

# Download the WordNet data

warnings.filterwarnings('ignore')
vectorizer=joblib.load(r"D:\machine learning project\vectorizer.pkl")
le = joblib.load(r"D:\machine learning project\le.pkl")
linear_svc=joblib.load(r"D:\machine learning project\linear_svc.pkl")

#function to create clean data
def clean_data(d):
    d = d.lower()
    a=[]
    #         removing links from text data
    d = re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+', ' ', d)

    #         removing other symbols
    d = re.sub('[^0-9a-z]', ' ', d)
    for i in tk.tokenize(d):
        if i not in sw and len(i)>3:
            a.append(i)
    return a
def lemmatize(a):
    k=[lematizer.lemmatize(i) for i in a]
    lm_sentence=[" ".join(k)]
    return lm_sentence
def vector(sentence):
    new=vectorizer.transform(sentence).toarray()
    return new
def display_about_page():
    st.title("About the Project")
    st.write(
        "Welcome to the project overview page! Here, we provide details about the machine learning model, its accuracy, the dataset used, and the societal impact of this project."
    )

    # Display details about the machine learning model
    st.header("Machine Learning Model")
    st.write(
        "Our project is designed to predict MBTI (Myers-Briggs Type Indicator) personalities based on users' text input about themselves. We utilize a Linear SVC (Support Vector Classifier) machine learning model for personality prediction."
    )

    # Display accuracy information
    st.header("Model Accuracy")
    st.write(
        "The Linear SVC model achieves an accuracy score of 0.67 on our dataset. This accuracy reflects the model's ability to correctly predict MBTI personalities based on user text input."
    )

    # Display information about the dataset used
    st.header("Dataset Used")
    st.write(
        "The machine learning model is trained on a diverse dataset of user text inputs, with labeled MBTI personalities. This dataset enables the model to learn patterns and relationships between text features and personality types."
    )

    # Display societal impact
    st.header("Societal Impact")
    st.write(
        "Our project contributes to understanding and exploring the connection between language patterns and personality traits. By predicting MBTI personalities from text input, we aim to provide individuals with insights into their communication styles and preferences. This understanding can be valuable for personal development, communication strategies, and fostering better interpersonal relationships."
    )

    # Display project developer information
    st.header("Project Developer")
    st.write(
        "This project was developed by Muhammed Rashid. If you have any questions or feedback, feel free to reach out at [insert contact information]."
    )


def display_personality_info(personality_type):
    st.header(f"Personality Type: {personality_type}")

    if personality_type == "ISTJ":
        st.write(""""
                    ISTJs, known as "The Inspectors," bring a meticulous and organized approach to everything they do. 
                    Their strong sense of duty and responsibility makes them reliable and steadfast individuals. While 
                    they may seem reserved initially, ISTJs have a dry and often unexpected sense of humor that emerges 
                    in comfortable settings. They thrive in roles that demand attention to detail, structured planning, 
                    and methodical execution. 
                    ISTJs are loyal friends and coworkers, and their commitment to order ensures that tasks are
                    completed efficiently. As pragmatists, they appreciate practical solutions and excel in 
                    environments where their systematic approach can be put to good use.
                    """"")

    elif personality_type == "ISFJ":
        st.write(""""
                    ISFJs, known as "The Protectors," embody warmth, empathy, and a genuine concern for others. Their 
                    nurturing and caretaking tendencies make them the pillars of support in both personal and professional 
                    spheres. ISFJs are meticulous planners, ensuring that every detail contributes to a harmonious 
                    environment. They express their creativity through practical means, often crafting thoughtful 
                    gestures that resonate with those around them. Loyalty is a hallmark of ISFJs, and their dedication 
                    to relationships shines through in their unwavering support. While they may shy away from the 
                    spotlight, their impact is profound, creating a nurturing atmosphere where individuals can flourish 
                    and feel valued.
                    """"")


    # Continue with detailed explanations for other personality types...

    elif personality_type == "INFJ":
        st.write("""
                    INFJs, known as "The Counselors," possess a unique blend of insight, creativity, and compassion. 
                    They are often driven by a profound sense of purpose, seeking to make a positive impact on the world. 
                    INFJs are deeply attuned to the needs and emotions of others, making them 
                    empathetic and understanding individuals. While they may appear reserved, their passion for 
                    meaningful connections and their creative endeavors reveal itself in their pursuits. Often found in 
                    helping professions, INFJs channel their intuition and empathy to guide others toward self-discovery
                    and personal growth. Their idealism and vision contribute to a world enriched by 
                    compassion and understanding.
                    """)
    elif personality_type == "INTJ":
        st.write("""
                    INTJs, known as "The Masterminds," are strategic and analytical thinkers with a forward-thinking 
                    vision. Their ability to solve complex problems and navigate uncertainty sets them 
                    apart as visionary leaders. Driven by a strong sense of purpose, INTJs excel in roles that 
                    allow them to shape the future through innovative ideas and strategic insights. While their rational
                    approach may make them seem reserved, INTJs have a playful and creative side that emerges in moments
                    of inspiration. Their pursuit of efficiency and autonomy leads them to explore 
                    uncharted territories, making them catalysts for change in both professional and personal spheres.
                    """)
    elif personality_type == "ISTP":
        st.write("""
                    ISTPs, known as "The Craftsmen," are pragmatic and hands-on problem solvers. They thrive in dynamic 
                    environments, utilizing their logical thinking to troubleshoot and innovate. ISTPs are adaptable and 
                    resourceful, often navigating challenges with a calm and collected demeanor. While they may seem 
                    reserved, their love for action and exploration reveals itself in their enthusiasm for hands-on 
                    experiences. ISTPs excel in fields that require practical application of skills and a knack
                    for finding solutions in the midst of uncertainty. Their keen observational skills and ability 
                    to think on their feet make them valuable assets in any situation that demands 
                    a quick and effective response.
                    """)
    elif personality_type == "ISFP":
        st.write("""
                    ISFPs, known as "The Composers," embody a free-spirited and artistic approach to life. 
                    They have a deep appreciation for aesthetics and express themselves through creative outlets 
                    like art, music, or other forms of self-expression. ISFPs are often reserved, but their authenticity
                    and individuality shine through in their actions and creations. They thrive in environments 
                    that allow them to explore their creative impulses and bring a unique perspective to their endeavors
                    While they may not seek the spotlight, ISFPs contribute to the world by infusing it
                    with beauty, emotion, and a touch of whimsy, leaving an indelible mark on those who experience
                     their creations.
                    """)
    elif personality_type == "INFP":
        st.write("""
                    INFPs, known as "The Healers," are dreamers with a profound sense of empathy and idealism.
                    They are driven by strong personal beliefs and values, seeking to forge connections that transcend 
                    superficial interactions. INFPs channel their creativity and compassion into making a positive 
                    impact on the world around them. While they may appear reserved, their inner world is rich with
                    imaginative ideas and a deep well of emotions. INFPs are often advocates for social causes, using
                    their unique blend of creativity and empathy to inspire change. Their authenticity and commitment
                    to their ideals create a ripple effect, encouraging others to embrace their true selves and 
                    contribute to a more compassionate world.
                    """)
    elif personality_type == "INTP":
        st.write("""
                    INTPs, known as "The Architects," are intellectual explorers with a passion for understanding the 
                    complexities of the world. They thrive in abstract thinking and are innovative problem-solvers. 
                    While appearing reserved, INTPs have a playful and quirky side that emerges when they are engaged in 
                    pursuits that stimulate their intellectual curiosity. They value independence and intellectual 
                    freedom, often immersing themselves in the pursuit of knowledge. INTPs contribute groundbreaking 
                    ideas and perspectives, shaping industries and disciplines with their logical and inventive thinking
                    Their ability to envision possibilities and question established norms makes them indispensable in 
                    scenarios that require creative and unconventional solutions.
                    """)
    elif personality_type == "ESTP":
        st.write("""
                    ESTPs, known as "The Dynamos," are energetic and action-oriented individuals
                    who thrive on challenges. They excel in high-pressure situations and have a natural talent 
                    for thinking on their feet. ESTPs are charismatic leaders who inspire and energize those around them
                    They enjoy taking calculated risks and navigating dynamic environments with ease. While their focus 
                    on the present moment may make them appear spontaneous, ESTPs possess a practical and
                    results-oriented mindset. Their resourcefulness and adaptability make them invaluable in situations 
                    that demand quick decision-making and effective problem-solving.
                    """)
    elif personality_type == "ESFP":
        st.write("""
                    ESFPs, known as "The Performers," are life enthusiasts who bring energy and vivacity to any setting.
                    They live for the moment, embracing new experiences with zest and enthusiasm. ESFPs are often 
                    the life of the party, captivating others with their magnetic personality.They have a natural talent
                    for creating joyous atmospheres and thrive in social interactions. While they may appear spontaneous
                    ESFPs have a practical side that allows them to navigate life with
                    a blend of excitement and groundedness. Their ability to infuse joy into everyday experiences and 
                    connect with people on a personal level makes them cherished companions in both personal and 
                    professional circles.
                    """)
    elif personality_type == "ENFP":
        st.write("""
                    ENFPs, known as "The Champions," are enthusiastic and imaginative individuals who constantly 
                    seek inspiration.They are passionate advocates for their beliefs and value personal connections 
                    deeply.ENFPs channel their creativity and optimism into making a positive impact on the world around
                    them. While their energy and excitement are contagious, ENFPs also possess a reflective side, 
                    contemplating the deeper meanings of life. They thrive in environments that allow them 
                    to explore possibilities and contribute to meaningful causes. ENFPs are influential leaders and 
                    collaborators, inspiring others to embrace their passions and envision a future filled 
                    with innovation and positive change.
                    """)
    elif personality_type == "ENTP":
        st.write("""
                     ENTPs, known as "The Visionaries," are innovative thinkers who thrive on intellectual challenges. 
                    They are naturally curious and adaptable, exploring new ideas with a relentless pursuit of knowledge
                     ENTPs are often seen as creative idea generators who enjoy pushing the boundaries of conventional 
                     thinking. While their playful and exploratory nature may make them seem unconventional, 
                     ENTPs possess a strategic mind that allows them to navigate complexities with ease. They contribute
                      to progress and innovation by questioning the status quo and envisioning possibilities that others
                    may overlook. ENTPs are catalysts for change, injecting fresh perspectives into a variety of fields 
                    and endeavors.
                    """)
    elif personality_type == "ESTJ":
        st.write("""
                    ESTJs, known as "The Supervisors," are authoritative leaders with a focus on 
                    efficiency and structure. They thrive in roles that demand strong organizational skills and decisive
                    decision-making. While they may seem stern, ESTJs are warm and supportive mentors who guide others 
                    toward success. Their practical approach to problem-solving and emphasis on tradition make them 
                    dependable pillars in both personal and professional realms. ESTJs value loyalty and reliability, 
                    fostering a sense of camaraderie in their social circles. They contribute to environments 
                    that benefit from order and structure,ensuring that tasks are completed with precision and attention
                     to detail.
                    """)
    elif personality_type == "ESFJ":
        st.write("""
                    ESFJs, known as "The Providers," are sociable and caring individuals who prioritize 
                    the well-being of those around them. They excel in creating supportive environments and fostering 
                    connections with a genuine and nurturing approach. ESFJs are excellent communicators, using their 
                    natural charisma to mediate conflicts and bring people together. While their focus on the needs of 
                    others may make them seem selfless, ESFJs also appreciate the joy that comes from shared experiences 
                    Their warmth and empathy create a sense of community, making them cherished friends and confidantes 
                    who contribute to the emotional well-being of those around them.
                       """)
    elif personality_type == "ENFJ":
        st.write("""
                    ENFJs, known as "The Teachers," are charismatic leaders with a passion for motivating and inspiring 
                    others. They thrive in roles that allow them to make a positive impact on people's lives, 
                    often guiding individuals toward personal and professional growth. ENFJs are 
                    intuitive and empathetic, creating an atmosphere of trust and collaboration. While they may take on 
                    the responsibility of leadership, ENFJs also value collaboration and collective success. 
                    Their ability to understand the emotions and needs of others makes them effective communicators and 
                    mentors. ENFJs contribute to environments where individuals feel valued and empowered, 
                    fostering a sense of purpose and camaraderie.
                       """)
    elif personality_type == "ENTJ":
        st.write("""
                    ENTJs, known as "The Commanders," are assertive and strategic leaders who thrive in 
                    positions of authority. They are driven by a desire for efficiency and progress, 
                    often seeking opportunities for growth and improvement. ENTJs possess a decisive nature and
                    a visionary thinking style that allows them to set ambitious goals. While they may appear focused on
                    the big picture, ENTJs are also attentive to the details that contribute to overall success. 
                    Their leadership style is characterized by assertiveness and a commitment to achieving results. 
                    ENTJs excel in dynamic environments where they can implement their strategic vision and guide others
                     toward success.
                       """)

def main():

    st.title("MBTI and 16 Personalities")
    st.write("Welcome to our website! Learn about MBTI and explore each of the 16 personalities.")

    # Sidebar for navigation
    navigation = st.sidebar.radio("Navigation", ["Home",  "Personalities","Find your Personality type","About"])

    # Home page
    if navigation == "Home":
        st.header("Understanding MBTI - An Introduction")
        st.write("""The Myers-Briggs Type Indicator (MBTI) is a widely utilized personality assessment tool, 
                    drawing from Carl Jung's theories. It classifies individuals into 16 personality types, 
                    each denoted by a four-letter code reflecting preferences in four dichotomies: 
                    Extraversion/Introversion, Sensing/Intuition, Thinking/Feeling, and Judging/Perceiving. 
                    This tool aims to offer insights into natural inclinations, decision-making approaches, 
                    and interpersonal behaviors. Often employed in career counseling and team development, 
                    the MBTI provides a framework for understanding diverse personality traits. However, 
                    it has been critiqued for lacking scientific validation and oversimplifying the 
                    intricacies of human personality. While acknowledging its popularity, users are 
                    encouraged to interpret results cautiously, recognizing the MBTI as a tool for 
                    self-reflection rather than an exhaustive classification system.""")



    # About MBTI page
    elif navigation == "Personalities":
        st.header("Exploring the 16 Personality Types")
        personality_types = [" ",'INTJ', 'ENTP', 'INFJ', 'ENFP', 'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
                             'ISTP', 'ISFP', 'ESTP', 'ESFP', 'INTP', 'INFP', 'ENTJ', 'ENFJ']

        selected_type = st.selectbox("Select a Personality Type", personality_types)
        if selected_type==" ":
            st.write(" ")
        else:
            display_personality_info(selected_type)
            st.write("""
                    These extended explanations provide a more comprehensive understanding of the unique qualities and 
                    contributions of each personality type. It's essential to recognize the diversity within each type 
                    and appreciate the richness they bring to various aspects of life and work. From the meticulous and 
                    organized approach of ISTJs to the free-spirited and artistic nature of ISFPs, each personality type
                     contributes distinct strengths and perspectives. Understanding these differences fosters better 
                     communication, collaboration, and appreciation for the varied ways individuals approach challenges 
                     and relationships. Embracing the complexity and depth of each personality type enriches 
                     our interactions, creating a more inclusive and harmonious environment where everyone's 
                     unique qualities are valued and celebrated.
                    """)

    elif navigation=="Find your Personality type":
        st.header("lets find your  Personalty type")
        st.write("Hey, spill the tea! What's your story? Tell me about the amazing person behind the scenes. "
                 "What passions fuel your fire, and what do you love diving into when you're not conquering the world? "
                 "Let's get to know the real you! ")
        d = st.text_area("Enter text here", "")
        Cloud_button = st.button("Word Cloud")

        if Cloud_button:
            cleaned_data=clean_data(d)
            st.write(f"no of words present in your text:{len(cleaned_data)}")
            lematized_data = lemmatize(cleaned_data)

            # Generate Word Cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(lematized_data[0])

            # Display the Word Cloud using matplotlib
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')  # Turn off the axis labels
            st.pyplot(fig)
        person_button = st.button("Personality")
        if person_button:
            cleaned_data = clean_data(d)
            st.write(f"no of words present in your text:{len(cleaned_data)}")
            lematized_data = lemmatize(cleaned_data)
            new = vector(lematized_data)
            Y_new = linear_svc.predict(new)
            Y = le.inverse_transform(Y_new)
            display_personality_info(Y[0])
    elif navigation == "About":
        display_about_page()








if __name__ == "__main__":
    main()

