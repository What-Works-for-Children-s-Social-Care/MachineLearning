### Text functions for WWCSC Predictive Analytics Project
from functools import reduce
import glob
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Token
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS

# Used in 1a
def aggregate_previous_values(func, df = pd.DataFrame(), id_to_group_by = str, columns = []):
    '''
    func: 'sum' or 'mean' or 'concat' (for text)
    df: dataframe you'd like to aggregate the previous data for (values need to be sorted by id_to_group_by and Date)
    id_to_group_by: ID which identifies the same individual over time
    columns: all those which you'd like to aggregate through that function
    returns dataframe with both the columns including and excluding the current data (kept for comparison)
    '''
    # Takes the expanding mean 
    if func == 'sum':
        df_agg = df.groupby(by = [id_to_group_by], as_index = True)[columns].expanding().sum().reset_index(level = 0, col_level = 0, drop = False)
        columns_shifted = ['previous_exc_current_sum_'+col for col in columns]
        columns_renamed = ['previous_inc_current_sum_'+col for col in columns]
    if func == 'mean':
        df_agg = df.groupby(by = [id_to_group_by], as_index = True)[columns].expanding().mean().reset_index(level = 0, col_level = 0, drop = False)
        columns_shifted = ['previous_exc_current_mean_'+col for col in columns]
        columns_renamed = ['previous_inc_current_mean_'+col for col in columns]    
    if func == 'concat':
        df_agg = df.groupby(by = [id_to_group_by], as_index = True)[columns].expanding().apply(lambda row: ' '.join(row.values.astype(str)), axis=1).reset_index(level = 0, col_level = 0, drop = False)
        columns_shifted = ['previous_exc_current_concatenated_'+col for col in columns]
        columns_renamed = ['previous_inc_current_concatenated_'+col for col in columns]       
    # Shifts aggregations downwards to avoid including current observation
    df_agg[columns_shifted] = df_agg.groupby(by = [id_to_group_by])[columns].shift(periods=1, axis=0, fill_value=0)
    df_agg.rename(columns = dict(zip(columns, columns_renamed)), inplace = True)
    return(df_agg)


# Used in 2 
def names_w_placeholders(rq1_text):
    '''Replace available names with placeholders'''
    # Create counters for number of names replaced, number of documents with replaced names
    documents_names_n, names_total_n = 0, 0
    for text_col in text_columns:
        already_done_list = []
        for idx in rq1_text.index:
            print(idx)
            string = rq1_text.loc[idx, text_col]
            #print(string)
            if (string.strip() != '') and (idx not in already_done_list):
                # Dealing with formatting issues
                string = re.sub("{[a-z:]*}", "", str(string))
                # Strip space characters
                string = re.sub(r'\W+', ' ', str(string))
                names_n = 0
                for name in names:
                    #print(name)
                    name_at_idx = rq1_text.loc[idx,name]
                    if name_at_idx == name_at_idx:
                        for n in name_at_idx:
                            print(n)
                            if n.strip() != '':
                                stringn = re.subn(r"\b{}\b".format(n), name, str(string), flags = re.IGNORECASE)
                                string = stringn[0]
                                names_n += stringn[1]
                #print("Names replaced in document: ", names_n)
                repeated_text_indices = list(np.where(rq1_text[text_col] ==  rq1_text.loc[idx,text_col])[0])
                #print(repeated_text_indices)
                if repeated_text_indices:
                    print(repeated_text_indices)
                    rq1_text.at[repeated_text_indices,text_col] = string
                    already_done_list.extend(repeated_text_indices)
                    if names_n != 0:
                        documents_names_n += len(repeated_text_indices)
                        names_total_n += len(repeated_text_indices)*names_n
                else:
                    rq1_text.at[idx,text_col] = string
                    if names_n != 0:
                        documents_names_n += 1
                        names_total_n += names_n
                #print("Number of documents with names replaced: ", documents_names_n)
                #print("Number of names replaced total: ",names_total_n)
        return (rq1_text, documents_names_n, names_total_n)

# Used in 3
def text_feature_creation(df, column, emotions, nlp, emolex_words, concreteness_df):
    '''Function to create features relating to polarity, subjectivity, 
    emotions and concreteness'''
    if df[column].isna().sum() != len(df[column]):
        new_df = pd.DataFrame(0, index=df.index, columns=['polarity', 'subjectivity'], dtype = 'object')
        emo_df = pd.DataFrame(0, index=df.index, columns=emotions)
        concrete_df = pd.DataFrame(0, index=df.index, columns=['concreteness'])
        already_done_list = []
        for i, row in df.iterrows():
            print("Index: ", i)
            if i not in already_done_list:
                text = df.loc[i,column]
                if text == text:
                    # Correct spelling after anonymisation
                    # Add social work specific words to avoid correcting those?
                    try:
                        #print("Correcting spelling")
                        blob = TextBlob(text)
                        #blob = str(blob.correct()) 
                        # Sentiment
                        polarity = blob.sentiment.polarity
                        #print("Polarity: ", polarity)
                        new_df.at[i,'polarity'] = polarity
                        subjectivity = blob.sentiment.subjectivity
                        #print("Subjectivity: ", subjectivity)
                        new_df.at[i, 'subjectivity'] = subjectivity
                    except(TypeError):
                        print("TypeError")
                        pass
                    doc = nlp(df.loc[i,column])
                    for token in doc:
                        emo_score = emolex_words[emolex_words.word == token.lemma_]
                        if not emo_score.empty:
                            for emotion in list(emotions):
                                emo_df.at[i, emotion] += emo_score[emotion]
                        #print("Emo_df: ", emo_df)
                        concreteness_score = concreteness_df.loc[concreteness_df.Word == token.lemma_, "Conc.M"]
                        if not concrete_df.empty and len(concreteness_score) != 0:
                            concrete_df.at[i,'concreteness'] += float(concreteness_score)
                    # Standardise by dividing by length of the document (polarity and subjectivity are already 0-1)
                    print("Length of document: ", len(df.loc[i,column]))
                    concrete_df.loc[i,'concreteness'] = concrete_df.loc[i,'concreteness'] / len(df.loc[i,column])
                    emo_df.at[i,"Length of document"] = len(df.loc[i,column])
                    # If already processed, use processed data
                    repeated_text_indices = df.index[np.where(df[column] ==  df.loc[i,column])]
                    print("Repeated text indices: ", repeated_text_indices)
                    if repeated_text_indices != []:
                        new_df.loc[repeated_text_indices,'polarity'] = polarity
                        new_df.loc[repeated_text_indices,'subjectivity'] = subjectivity
                        emo_df_repeated = pd.concat([pd.DataFrame(emo_df.loc[i,]).T]*len(repeated_text_indices))
                        emo_df_repeated.set_index(repeated_text_indices, inplace = True)
                        emo_df.update(emo_df_repeated)
                        concrete_df_repeated = pd.concat([pd.DataFrame(concrete_df.loc[i,]).T]*len(repeated_text_indices))
                        concrete_df_repeated.set_index(repeated_text_indices, inplace = True)
                        concrete_df.update(concrete_df_repeated)
                        already_done_list.extend(repeated_text_indices)
            # Standardise by length of document (divide whole dataframe rather than row by row)
        emo_df2 = emo_df.div(emo_df["Length of document"].values, axis = 0)
        emo_df2.drop(columns = ["Length of document"], inplace = True)
        # Each column will have its own features so need to rename
        new_df = new_df.add_prefix(column + '_')
        emo_df2 = emo_df2.add_prefix(column + '_')
        concrete_df = concrete_df.add_prefix(column + '_')
        new_df = pd.concat([new_df, emo_df2, concrete_df], axis=1)
        print(new_df)
        return new_df

    
def check_anonymisation_stage_is_complete(file_stub, file_path, text_columns, n_rows, save_every):
    '''Compare saved anonymised files to the files expected if successful anonymisation of all files.
    Helpful when you've had to stop and start anonymisations.'''
    full_file_stub = file_path + '\\' + file_stub

    files = [file for file in glob.glob("{}_*.pkl".format(str(full_file_stub)))]
    
    def roundint(value, base=500):
        return int(value) - int(value) % int(base)
    
    n_rows_range = roundint(n_rows, save_every)
    
    # Create list of existing file names associated with that anonymisation stage
    df_list = []
    for file in files:
        filename = open(file, "rb")
        file_n = re.sub(file_path, "", file) 
        file_n = re.sub(r".pkl", "", file_n)
        df_list.append(file_n)

    # Create list of expected file names
    ideal_file_names = []
    for col in text_columns:
        j = 0
        for i in range(save_every, n_rows_range + save_every, save_every):
            if j==0:
                ideal_file_names.append('\\' + file_stub + '_' + str(j) + '_' + str(i) + '_' + col)
            else:
                ideal_file_names.append('\\' + file_stub + '_' + str(j+1) + '_' + str(i) + '_' + col)
            j = i
        # Make sure I have the bit at the end
        ideal_file_names.append('\\' + file_stub + '_' + str(n_rows_range+1) + '_' + str(n_rows) + '_' + col)  
    
    # Returns missing file names
    return (set(ideal_file_names) - set(df_list), ideal_file_names)   

# Call in data (names replaced with placeholders)
def knit_together_anonymised_pickled_files(file_stub, file_path, ideal_file_list, text_columns):
    '''Knit together all the saved files from this stage of the anonymisation.
    Returns a single dataframe. Dataframe shape should match the shape of the rq dataframe.
    ideal_file_names comes from check_anonymisation_stage_is_complete above - otherwise other
    files which may be misspellings but still fit the pattern will be incorporated.'''
    file_stub_full = file_path + '\\' + file_stub
    files = [file for file in glob.glob("{}_*.pkl".format(file_stub_full))]

    df_dict = {}
    for file in files:
        filename = open(file, "rb")
        files_short = pickle.load(filename)
        file_n = re.sub(file_path, "", file) 
        file_n = re.sub(r".pkl", "", file_n)
        if file_n in ideal_file_list:
            df_dict[file_n] = files_short

    # Concatenate rows in the same column
    col_wo_identifier_dict = {}
    for text_col in text_columns:
        print(text_col)
        text_col_df = [df for df in df_dict.keys() if text_col in df]
        if '_previous' in text_col:
            text_col_df = [df for df in text_col_df if '_previous' in df]
        else:
            text_col_df = [df for df in text_col_df if '_previous' not in df]
        if 'EH_' in text_col:
            text_col_df = [df for df in text_col_df if 'EH_' in df]
        else:
            text_col_df = [df for df in text_col_df if 'EH_' not in df]    
        text_col_df_values = [v for k, v in df_dict.items() if k in text_col_df]
        print(len(text_col_df_values))
        col_wo_identifier_dict[text_col] = pd.concat(text_col_df_values, axis = 0)
        col_wo_identifier_dict[text_col].sort_index(inplace = True)

        # Merge columns together (commented out because keys were all missing and merging on them was causing
        # a memory error. For the moment, I just concatenate the values outside the function
        #df_wo_identifier_together = reduce(lambda left, right: pd.merge(left,right,on=['Case Number', 'Contact Date'], how = 'left'), col_wo_identifier_dict.values())

    #return(df_wo_identifier_together)
    return(col_wo_identifier_dict)


# Used in 3
class RecogniseTagPhrases(object):
    '''class to make vulnerabilities a tag in spacy pre-processing'''
    def __init__(self, nlp, phrases, tag):
        self.phrases = phrases
        self.tag = tag
        # Register a new token extension to flag phrase
        Token.set_extension(self.tag, default=False, force = True)
        self.matcher = PhraseMatcher(nlp.vocab)
        for phrase in self.phrases:
            self.matcher.add(phrase, None, nlp(phrase))

    def __call__(self, doc):
        # This method is invoked when the component is called on a Doc
        matches = self.matcher(doc)
        spans = []  # Collect the matched spans here
        for match_id, start, end in matches:
            spans.append(doc[start:end])
        with doc.retokenize() as retokenizer:
            for span in spans:
                try:
                    retokenizer.merge(span)
                    for token in span:
                        token._.set(self.tag, True) 
                except(ValueError):
                    continue
        return doc
    
def topic_top_words(model, feature_names, n_top_words, avoid_list):
    topic_top_words_dict = {}
    for i, topic in enumerate(model.components_):
        #print(i)
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1] if feature_names[i] not in avoid_list]
        freq_word_topic = [topic[i] for i in topic.argsort()[:-n_top_words - 1:-1] if feature_names[i] not in avoid_list] # no. of times word was assigned to topic
        topic_top_words_dict[i] = list(zip(top_words, freq_word_topic))
    return topic_top_words_dict   

def create_wordcloud(model, wordcloud_avoid_list, n_top_words, file_stub, col, vectorizer, num_topics):
    
    
    wwcsc_brand_colours = (['#4d4d51', '#ff7057', '#fd9b8c', '#fec6bd', '#88d0d9', '#c4e7ea', '#b0c7cc',
                           '#d0dde1', '#27523a', '#5b9b54', '#231f20', '#eef0f2'])

    cols = [color for i, color in enumerate(wwcsc_brand_colours)]

    cloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    lda = model['lda']
    if vectorizer == 'tfidf':
        model_vectorizer = model['tfidf']
    elif vectorizer == 'tf_vec':
        model_vectorizer = model['tf_vec']
    else:
        model_vectorizer = model['count_vec']
    feature_names = model_vectorizer.get_feature_names()
    topics = topic_top_words(lda, feature_names, n_top_words = n_top_words, avoid_list = wordcloud_avoid_list)
    
    if num_topics < 2:
        nrows = 1
    else:
        nrows = math.ceil(num_topics / 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(15,15), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig('../../Graphs/Word clouds {} {}.png'.format(file_stub, col), transparent=False, dpi=80, bbox_inches="tight")
    plt.show()
    
## Vulnerabilities


vulnerabilities_phrases_dict = ({"mental health": 
                                ["mental health", "self harm", "depression", "anxiety", "depressed", "anxious", "suicidal"],
                            "CSE": ["child sexual exploitation", "CSE", "sex offender"],
                                "CCE": ["child criminal exploitation", "CCE", "gang"],
                           "forced marriage": ["forced marriage"], 
                                "troubled families": ["troubled families"], 
                                "poor inter-parental relationship": ["domestic abuse", "divorce", "divorced", "separation", 
                                 "separate", "separated"], 
                                "low income": ["low income", "struggling financially"],
                            "single parent": ["single parent", "single mother", "single father", "single mum", "single dad"],
                            "unemployed": ["unemployed", "workless", "NEET"],
                            "food poverty": ["food poverty", "hungry", "not enough food"], 
                            "prison": ["prison", "offender", "bail", "probation"],
                            "FSM": ["free school meals", "FSM"],
                            "homeless or housing issues": ["homeless", "insecure accommodation", 
                            "unstable accommodation", "insecure housing", "unstable housing", "insecure tenancy", "hostel"],
                            "substance abuse": ["addiction", "addicted", "alcohol", "cannabis", "heroin", "weed","cocaine", 
                            "substance abuse", "drug user"], 
                            "young carer": ["young carer"], 
                            "teenage parent": ["teenage parent"], 
                            "excluded": ["excluded", "exclusion"],
                             "missing": ["missing"], 
                            "alternative education": ["pupil referral unit", "PRU"], 
                           "abuse or neglect": ["emotional abuse", "neglect", "maltreatment", "physical abuse"],
                            "victim of crime": ["victim of crime"],
                            "trafficked": ["trafficking", "trafficked"],
                           "FGM": ["female genital mutilation", "FGM"], 
                            "seeking asylum or immigration issues": ["immigration detention", "undocumented", "asylum"],
                            "secure institution": ["secure institution", "children's home", "semi-independent living",
                            "secure welfare accommodation", "secure welfare detention"], 
                            "disabilty or long term illness": ["disabled","disability", "special educational need", "SEN", "SEND", 
                            "life-limiting illness", "autism", "speech difficulties", "language difficulties",
                            "communication difficulties"],
                            "radicalised": ["radicalised","channel support", "prevent programme"]})
