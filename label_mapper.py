# Synonym generation task.
# Lets consider task with aim of profiling cryptocurrency influencers in social media.  the Low-resource influencer profiling is one of tasks with the following classes.
# Generate 10 most appropiate synonyms for each class and format the output in a dictionary with {`class`:`synonyms`} format:
# Classes: 
# ```
#  (1) null, (2) nano, (3) micro, (4) macro, (5) mega
# ```
SUBTASK_1_Verbalizer = {
    0: ["no influencer", "zero influencer", "zero", "none", "nonexistent influencer", 
                      "empty", "empty influencer", "void","lack of influencer",  "absent", "absent influencer"],
    1: ["nano", "tiny", "minuscule", "small", "petite", "microscopic", "modest"],
    2: ["micro", "small-scale",  "miniature", "minor", "minute", "slight", "limited", "mild"],
    3: ["macro", "large-scale", "large", "massive", "huge", "giant", "enormous"],
    4: ["mega", "gigantic", "colossal", "immense", "vast", "monumental"]
}
SUBTASK_1_LABEL2ID = {'no influencer': 0, 'nano': 1, 'micro': 2, 'macro': 3, 'mega': 4}
SUBTASK_1_ID2LABEL = {0: 'no influencer', 1:'nano', 2:'micro', 3:'macro', 4:'mega'}
SUBTASK_1 = {
    "no influencer": ["no influencer", "zero influencer", "zero", "none", "nonexistent influencer", 
                      "empty", "empty influencer", "void","lack of influencer",  "absent", "absent influencer"],
    "nano": ["nano", "tiny", "minuscule", "small", "petite", "microscopic", "modest"],
    "micro": ["micro", "small-scale",  "miniature", "minor", "minute", "slight", "limited", "mild"],
    "macro": ["macro", "large-scale", "large", "massive", "huge", "giant", "enormous"],
    "mega": ["mega", "gigantic", "colossal", "immense", "vast", "monumental"]
}




# Synonym generation task for prompt based text classification task
# Task Definition:
# The aim is to identify cryptocurrency influencers interest in the Twitter
# Given a couple of tweets from a user, identify that user interested in which of the following: `technical information`, `price update`,  `trading matters`,  `gaming`, and `other`
# Generate 5 synonyms for each class and format the output in a dictionary with {`class`:`synonyms`} format:
# Classes: 
# ```
#  `technical information`, `price update`,  `trading matters`,  `gaming`, and `other`
# ```
SUBTASK_2 = {
  "technical information": ["technical information", "crypto insights", "blockchain details", "technology updates", "digital currency knowledge",  "cryptocurrency analysis"],
  "price update": ["price update", "market movements", "crypto price changes", "value fluctuations", "exchange rate updates", "price fluctuations"],
  "trading matters": ["trading matters", "investment strategies", "trading tips", "market analysis", "investment opportunities", "trading insights"],
  "gaming": ["gaming", "cryptogaming", "blockchain gaming", "crypto entertainment", "decentralized gaming", "virtual currency games"],
  "other": ["other", "miscellaneous", "diverse interests", "non-specific", "general topics", "varied subjects"]
}
SUBTASK_2_LABEL2ID = {"technical information":0, "price update":1, "trading matters":2, "gaming":3, "other":4}





# Synonym generation task for prompt based text classification task
# Task Definition:
# The aim is to identify cryptocurrency influencers intent or purpose in the Twitter
# Given a couple of tweets from a user, identify that user purpose is which of the following:  `subjective opinion`, `financial information`,  `advertising`,  `announcement`
# Generate 5 synonyms for each class and format the output in a dictionary with {`class`:`synonyms`} format:
# Classes: 
# ```
#  `subjective opinion`, `financial information`,  `advertising`,  `announcement`
# ```
SUBTASK_3 = {
  "subjective opinion": ["subjective opinion", "personal viewpoint", "individual perspective", "subjective stance", "personal belief", "opinionated view"],
  "financial information": ["financial information", "economic data", "monetary details", "financial insights", "money-related information", "fiscal updates"],
  "advertising": ["advertising", "promotion", "marketing", "branding", "advertisement", "promotional activities"],
  "announcement": ["announcement", "declaration", "statement", "notice", "public disclosure", "proclamation"]
}
SUBTASK_3_LABEL2ID={"subjective opinion":0, "financial information":1, "advertising":2, "announcement":3}
