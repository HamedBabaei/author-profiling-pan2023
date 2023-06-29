

SUBTASK_1 = {
    "train":
        ["""Identify cryptocurrency influencers profiles from given tweets: \n\nTweets: {tweets} """,                                             #1
        """User tweets: "{tweets}" \n\nQuestion: What is the profile of this user in twitter?""",                                                 #2
        """{tweets} \n\nWhat profile is appropriate for this user from a cryptocurrency perspective?""",                                          #3
        """{tweets} \n\nIs this a cryptocurrency influencers?""",                                                                                 #4
        """Given collection of tweets from a user: "{tweets}"\n\nWhat is the user profile as a cryptocurrency influencers?""",                    #5
        """What is the user related aspect of the influencer using the following tweets??\n\nTweets:{tweets}""",                                  #6
        """Given the following user tweets, determine the profile of this user as a cryptocurrency influencer:\n\nUser tweets: {tweets}""",       #7
        """Consider the tweets provided: "{tweets}" \n\nWhat would be an appropriate profile for this user from a cryptocurrency perspective?""", #8
        """A user has posted the following collection of tweets: "{tweets}"\n\nWhat is the user's profile as a cryptocurrency influencer?""",     #9
        """Evaluate the given tweets to identify cryptocurrency influencers:\n\nTweets: {tweets} """                                              #10
    ],
    "test":"""Identify cryptocurrency influencers profiles from given tweets: \n\nTweets: {tweets}"""
}


SUBTASK_2 = {
    "train":[
        """Identify the user interest in cryptocurrency from the given tweets: \n\nTweets: {tweets}""",                                        #1
        """Analyze the given tweets to identify if the user has a particular interest in cryptocurrency. \n\nTweets:{tweets}""",               #2
        """User tweets: "{tweets}" \n\nQuestion: What is the user interest in cryptocurrency?""",                                              #3
        """Given collection of tweets from a user: "{tweets}"\n\nWhat is the user interest in cryptocurrency influencers?""",                  #4
        """{tweets} \n\nExamine the tweets and determine if the user exhibits an interest in cryptocurrency.""",                               #5
        """{tweets} \n\nFrom the provided tweets, ascertain whether the user shows interest in following or engaging with cryptocurrency?""",  #6
        """Evaluate the given tweets to identify the user's interest in cryptocurrency:\n\nTweets: {tweets}""",                                #7
        """Given the following user tweets, determine the user interest:\n\nUser tweets: {tweets}""",                                          #8
        """Consider the tweets provided: "{tweets}" \n\nIdentify the user interest?""",                                                        #9
        """A user has posted the following collection of tweets: "{tweets}"\n\nWhat is the user's preference in the cryptocurrency?""",        #10
    ],  
    "test":"""Identify cryptocurrency influencers interest from given tweets: \n\nTweets: {tweets} """
}

SUBTASK_3 = {
    "train":[
        """Identify the user intent in cryptocurrency from the given tweets: \n\nTweets: {tweets}""",                                        #1
        """Analyze the given tweets to identify if the user has a particular purpose in cryptocurrency. \n\nTweets:{tweets}""",              #2
        """User tweets: "{tweets}" \n\nQuestion: What is the user intent in cryptocurrency?""",                                              #3
        """Given collection of tweets from a user: "{tweets}"\n\nWhat is the user purpose in cryptocurrency influencers?""",                 #4
        """{tweets} \n\nExamine the tweets and determine if the user exhibits an intent in cryptocurrency.""",                               #5
        """{tweets} \n\nFrom the provided tweets, ascertain whether the user shows purpose in following or engaging with cryptocurrency?""", #6
        """Evaluate the given tweets to identify the user's intent in cryptocurrency:\n\nTweets: {tweets}""",                                #7
        """Given the following user tweets, determine the user aim in cryptocurrency:\n\nUser tweets: {tweets}""",                           #8
        """Consider the tweets provided: "{tweets}" \n\nIdentify the user intent?""",                                                        #9
        """A user has posted the following collection of tweets: "{tweets}"\n\nWhat is the user's goal in the cryptocurrency?""",            #10
    ],  
    "test":"""Identify the user intent in cryptocurrency from the given tweets: \n\nTweets: {tweets}""",
}

HYPOTHESIS_TEMPLATE = {
    1:"This user profile in cryptocurrency is a {}",
    2:"This influencer interest is a {}",
    3:"This influencer intent is a {}"
}