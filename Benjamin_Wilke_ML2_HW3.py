############################################################################################################################################
# Benjamin Wilke - Machine Learning II - Assignment 3 :: Decision Making with Matrices
############################################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
############################################################################################################################################
# Begin RestaurantRater Class Definition
############################################################################################################################################
class RestaurantRater:
    ## DEFINE RestaurantRater CLASS CONSTRUCTOR
    def __init__(self, people, restaurants):
        ## STORE RAW DICTIONARIES IN THE OBJECT CLASS FOR FUTURE FUNCTIONALITY
        self.__rawPeopleDict = people
        self.__rawRestaurantsDict = restaurants
        # CALCULATE AND STORE NUMBER OF PEOPLE AND NUMBER OF RESTARUANTS
        self.__peopleCount = len(list(self.__rawPeopleDict.keys()))
        self.__restaurantCount = len(list(self.__rawRestaurantsDict.keys()))
        ## INITIALIZE LISTS TO STORE KEYS/VALUES FROM PEOPLE AND RESTAURANTS (FORMED BELOW)
        self.__peopleKeys = []
        self.__peopleValues = []
        self.__restaurantsKeys = []
        self.__restaurantsValues = []
        self.__lastKey = 0
        ## POPULATE LISTS OF PEOPLE KEYS AND VALUES FOR MATRIX GENERATION
        for k1, v1 in people.items():
            row = []
            for k2, v2 in v1.items():
                self.__peopleKeys.append(k1+'_'+k2)
                if k1 == self.__lastKey:
                    row.append(v2)
                    self.__lastKey = k1
                else:
                    self.__peopleValues.append(row)
                    row.append(v2)
                    self.__lastKey = k1
        ## POPULATE LISTS OF RESTARUANT KEYS AND VALUES FOR MATRIX GENERATION
        for k1, v1 in restaurants.items():
            for k2, v2 in v1.items():
                self.__restaurantsKeys.append(k1+'_'+k2)
                self.__restaurantsValues.append(v2)
        ## INTITIALIZE AND POPULATE NUMPY MATRICES TO PERFORM MATRIX MULTIPLICATION / LINEAR COMBINATIONS
        self.__peopleMatrix = np.array(self.__peopleValues)
        self.__restaurantMatrix = np.reshape(self.__restaurantsValues, (self.__restaurantCount,4))

    ## RETURNS LIST OF PEOPLE
    def getPeople(self):
        return list(self.__rawPeopleDict.keys())
    ## RETURNS LIST OF RESTAURANTS
    def getRestaurants(self):
        return list(self.__rawRestaurantsDict.keys())
    ## RETURN MATRIX OF PEOPLE SHAPE
    def getPeopleMatrixShape(self):
        return self.__peopleMatrix.shape
    ## RETURN MATRIX OF RESTAURANT SHAPE
    def getRestaurantsMatrixShape(self):
        return self.__restaurantMatrix.shape
    ## RETURN PEOPLE MATRIX
    def getPeopleMatrix(self):
            return self.__peopleMatrix
    ## RETURN RESTAURANT MATRIX
    def getRestaurantsMatrix(self):
        return self.__restaurantMatrix
    ## RETURN FULL PEOPLExRESTAURANTS RESULTS
    def getPeopleDotRestaurantsRaw(self):
        return np.matmul(self.__peopleMatrix, self.__restaurantMatrix.T)
    ## RETURN SPECIFIC PERSONxRESTAURANTS RESULT
    def getPersonDotRestaurantsRaw(self, person):
        # simple error handling for passing name that doesn't exist
        try:
            index = list(self.__rawPeopleDict.keys()).index(person)
            return np.matmul(self.__peopleMatrix[index,],self.__restaurantMatrix.T)
        except ValueError:
            print("Name doesn't exist")
            return False
    ## RETURNS DESCENDING RANK ORDER OF ANY LENGTH LIST PROVDED IN PARAMETER (FOR EASY COMPARISON OF RANKS)
    def getRankedDesc(self, input_list):
        seq = sorted(input_list)
        index_asc = [seq.index(v) for v in input_list]
        index_inv = np.absolute(np.array(index_asc) - len(list(index_asc)))
        seq2 = sorted(index_inv)
        index_desc = [seq2.index(v) for v in index_inv]
        return np.array(index_desc) + 1
    ## RETURNS ASCENDING RANK ORDER OF ANY LENGTH LIST PROVDED IN PARAMETER (FOR EASY COMPARISON OF RANKS)
    def getRankedAsc(self, input_list):
        seq = sorted(input_list)
        index_asc = [seq.index(v) for v in input_list]
        return np.array(index_asc) + 1
    ## RETURN RANKING OF PEOPLExRESTAURANTS IN ASCENDING ORDER PER PERSON (WHERE HIGHEST SCORE IS 10)
    def getPeopleDotRestaurantsRankAsc(self):
        temp_list = []
        peopleDotRestaurantRaw = np.matmul(self.__peopleMatrix, self.__restaurantMatrix.T)
        for row in range(self.__peopleCount):
            temp_list.append(list(self.getRankedAsc(peopleDotRestaurantRaw[row])))
        return np.array(temp_list)
    ## RETURN RANKING OF PEOPLExRESTAURANTS IN DESCENDING ORDER PER PERSON (WHERE HIGHEST SCORE IS 1)
    def getPeopleDotRestaurantsRankDesc(self):
        temp_list = []
        peopleDotRestaurantRaw = np.matmul(self.__peopleMatrix, self.__restaurantMatrix.T)
        for row in range(self.__peopleCount):
            temp_list.append(list(self.getRankedDesc(peopleDotRestaurantRaw[row])))
        return np.array(temp_list)
    ## RETURN RAW COLUMN SUMS OF PEOPLExRESTAURANTS
    def getRestaurantRawScoreTotals(self):
        return np.sum(np.matmul(self.__peopleMatrix, self.__restaurantMatrix.T), axis=0)
    ## RETURN RANK COLUMN SUMS OF PEOPLExRESTAURANTS
    def getRestaurantRankScoreTotals(self):
        return np.sum(self.getPeopleDotRestaurantsRankAsc(), axis=0)
    ## OBJECT STR INVOCATION METHOD
    def __str__(self):
        return 'RestaurantRater : {0} People, {1} Restaurants'.format(self.__peopleCount, self.__restaurantCount)

############################################################################################################################################
# End RestaurantRater Class Definition - Begin Testing and Demonstration of Features
############################################################################################################################################

people = {"Jane": {"willingness to travel": 0.1596993,
                  "desire for new experience":0.67131344,
                  "cost":0.15006726,
                  "vegetarian": 0.01892,
                  },
          "Bob": {"willingness to travel": 0.63124581,
                  "desire for new experience":0.20269888,
                  "cost":0.01354308,
                  "vegetarian": 0.15251223,
                  },
          "Mary": {"willingness to travel": 0.49337138 ,
                  "desire for new experience": 0.41879654,
                  "cost": 0.05525843,
                  "vegetarian": 0.03257365,
                  },
          "Mike": {"willingness to travel": 0.08936756,
                  "desire for new experience": 0.14813813,
                  "cost": 0.43602425,
                  "vegetarian": 0.32647006,
                  },
          "Alice": {"willingness to travel": 0.05846052,
                  "desire for new experience": 0.6550466,
                  "cost": 0.1020457,
                  "vegetarian": 0.18444717,
                  },
          "Skip": {"willingness to travel": 0.08534087,
                  "desire for new experience": 0.20286902,
                  "cost": 0.49978215,
                  "vegetarian": 0.21200796,
                  },
          "Kira": {"willingness to travel": 0.14621567,
                  "desire for new experience": 0.08325185,
                  "cost": 0.59864525,
                  "vegetarian": 0.17188723,
                  },
          "Moe": {"willingness to travel": 0.05101531,
                  "desire for new experience": 0.03976796,
                  "cost": 0.06372092,
                  "vegetarian": 0.84549581,
                  },
          "Sara": {"willingness to travel": 0.18780828,
                  "desire for new experience": 0.59094026,
                  "cost": 0.08490399,
                  "vegetarian": 0.13634747,
                  },
          "Tom": {"willingness to travel": 0.77606127,
                  "desire for new experience": 0.06586204,
                  "cost": 0.14484121,
                  "vegetarian": 0.01323548,
                  }
}

restaurants  = {"Flacos":{"distance" : 2,
                        "novelty" : 3,
                        "cost": 4,
                        "vegetarian": 5
                        },
              "Joes":{"distance" : 5,
                        "novelty" : 1,
                        "cost": 5,
                        "vegetarian": 3
                      },
              "Poke":{"distance" : 4,
                        "novelty" : 2,
                        "cost": 4,
                        "vegetarian": 4
                      },
              "Sush-shi":{"distance" : 4,
                        "novelty" : 3,
                        "cost": 4,
                        "vegetarian": 4
                      },
              "Chick Fillet":{"distance" : 3,
                        "novelty" : 2,
                        "cost": 5,
                        "vegetarian": 5
                      },
              "Mackie Des":{"distance" : 2,
                        "novelty" : 3,
                        "cost": 4,
                        "vegetarian": 3
                      },
              "Michaels":{"distance" : 2,
                        "novelty" : 1,
                        "cost": 1,
                        "vegetarian": 5
                      },
              "Amaze":{"distance" : 3,
                        "novelty" : 5,
                        "cost": 2,
                        "vegetarian": 4
                      },
              "Kappa":{"distance" : 5,
                        "novelty" : 1,
                        "cost": 2,
                        "vegetarian": 3
                      },
              "Mu":{"distance" : 3,
                        "novelty" : 1,
                        "cost": 5,
                        "vegetarian": 3
                      }
}

myRestaurantRater = RestaurantRater(people, restaurants) # insantiate my RestaurantRater object with my people and restaurants data

############################################################################################################################################
# 1. Informally describe what a linear combination is and how it will relate to our resturant matrix.
# Linear combinations are critical for use in linear algebra. Essentially they are a set of terms each containing a constant that are
# multiplied and added together. These are common for the use in calculating cost for linear regression and/or forward feed calculations
# in neural networks.
############################################################################################################################################

############################################################################################################################################
# 2. Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent.
############################################################################################################################################
print("This is the linear combination for Tom:")
print(myRestaurantRater.getPersonDotRestaurantsRaw("Tom"))
print("This is the linear combination for Mike:")
print(myRestaurantRater.getPersonDotRestaurantsRaw("Mike"))
# Each of these entries represents Mike's preferences multipled by the corresponding attribute of the restuarant. This represents the
# "goodness of fit" between Mike's preferences and the attributes of the restaurant by observing the highest value.
#############################################################################################################################################

############################################################################################################################################
# 3. Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent?
############################################################################################################################################
print("This is the resulting linear combination for all people:")
print(myRestaurantRater.getPeopleDotRestaurantsRaw())
## also generate a Heatmap and save to file ####
ax = sns.heatmap(myRestaurantRater.getPeopleDotRestaurantsRaw(), annot=True)
ax.set(xlabel="Restaurants", ylabel="People", title="Heatmap of People to Restaurant Preferences")
ax.set_yticklabels(labels=myRestaurantRater.getPeople(), rotation=-45)
ax.set_xticklabels(labels=myRestaurantRater.getRestaurants(), rotation=90)
savefile = ax.get_figure()
savefile.tight_layout()
savefile.savefig("ML2_HW3_AllPeopleDotRestaurantsRaw_Heatmap.png")
print("Heatmap saved as: ML2_HW3_AllPeopleDotRestaurantsRaw_Heatmap.png")
# Each row of this matrix is the linear combination for a person (in the same way that Mike above was for a single person - you'll find
# Mike on the 4th entry). This represents the "goodness of fit" between each person's preferences and the attributes of the restaurant
# by observing the highest value per person.
#############################################################################################################################################

############################################################################################################################################
# 4. Sum all columns in M_usr_x_rest to get optimal restaurant for all users. What do the entry’s represent?
############################################################################################################################################
print("The restaurant order is:")
print(myRestaurantRater.getRestaurants())
print("The raw sum of each column (restaurant) is as follows:")
print(myRestaurantRater.getRestaurantRawScoreTotals())
print("Descending rank of raw sum of each column (restaurant) is as follows (highest scoring is 1):")
print(myRestaurantRater.getRankedDesc(myRestaurantRater.getRestaurantRawScoreTotals()))
# The entries represent the sum of each person by restaurant preferences. This is measuring the "best fit" restaurant considering each
# person's current preferences and corresponding restaurant attributes. The highest amount is the "best fit" restaurant accomodating all users.
############################################################################################################################################

############################################################################################################################################
# 5. Now convert each row in the M_usr_x_rest into a ranking for each user and  call it M_usr_x_rest_rank. Do the same as above to
# generate the optimal restaurant choice.
############################################################################################################################################
print("The ranking of each person's linear combinations in ascending order from raw score (so 10 is their highest combination):")
print(myRestaurantRater.getPeopleDotRestaurantsRankAsc())
print("The rank sum of each column (restaurant) is as follows:")
print(myRestaurantRater.getRestaurantRankScoreTotals())
print("Descending rank of rank sum of each column (restaurant) is as follows (highest scoring is 1):")
print(myRestaurantRater.getRankedDesc(myRestaurantRater.getRestaurantRankScoreTotals()))
############################################################################################################################################

############################################################################################################################################
# 6. Why is there a difference between the two?  What problem arrives?  What does represent in the real world?
# Descending rank of raw sum of each column (restaurant) is as follows:
# [ 5  6  4  1  3  7 10  2  9  8]
# Descending rank of rank sum of each column (restaurant) is as follows:
# [ 3  4  6  1  2  7 10  5  9  8]
# This issue has to do with the scale of the computed rankings. A person's computed raw rankings that are close to one another get very magnified
# when transformed to rankings. The real world impact is a person could be impartial to several choices, but now appear to care
# quite a bit! Lumping these people in with people who do have a preference across the same number of options is an unjust comparison.
############################################################################################################################################

############################################################################################################################################
# 7. How should you preprocess your data to remove this problem.
############################################################################################################################################
# Analysis should be done to understand how to compare people with similar computed rankings - this can be done with PCA as demonstrated later.
# We also have a problem with the restuarant ratings. A restaurant could have a score of 5 in all categories, which would heavily influence
# their presence in the scoring regardless of whether the person's preferences were for or against these attributes.
# We could impose requirements on how restaurants are scored.
pca_restaurants = PCA(n_components=2)
restaurants_matrix_PCA = pca_restaurants.fit_transform(myRestaurantRater.getRestaurantsMatrix())
print("PCA Restaurants Matrix 2 Principal Components")
print(pca_restaurants.components_)
print("PCA Restaurants Matrix explained variance:")
print(pca_restaurants.explained_variance_ratio_)

fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.scatter(restaurants_matrix_PCA[:,0], restaurants_matrix_PCA[:,1])
ax.set(xlabel="PC 1", ylabel="PC 2", title="PCA - People Matrix")
fig.tight_layout()
fig.savefig("ML2_HW3_RestaurantMatrixPCA.png")
#
# Nearly 75% of the variance in the restaurants is explained by PC1 and PC2. Plotting the restaurants shows a few clear outliers.
# These restaurants will have a very high impact on the resulting raw computed score.

############################################################################################################################################
# 8. Find user profiles that are problematic, explain why?
############################################################################################################################################
# User profiles could have a similar issue as restaurants. While the user preferences add up to 1, if they are almost evenly
# distributed (close to .25) then the user basically doesn't have any preferences (or at least care with any conviction about the 4 preferences
# that have been presented to them). This has the effect of the established restaurant's attributes affecting the scores more than the user.
#
# Basically the user's votes don't count.
#
# A simple test for this would be to examine each person's preferences to make sure they are not evenly distributed.
fig2, ax2 = plt.subplots(1,10, figsize=(12,12))
for each in range(myRestaurantRater.getPeopleMatrix().shape[0]):
    ax2[each].set(ylim=(0,1))
    for bar in range(myRestaurantRater.getPeopleMatrix().shape[1]):
        ax2[each].bar(bar, myRestaurantRater.getPeopleMatrix()[each,bar], .5)
        ax2[each].set(xlabel=myRestaurantRater.getPeople()[each])
        ax2[each].set_xticklabels([])
fig2.suptitle("Distribution of Person's Preferences")
fig2.legend(('Wilingness Travel', 'New Experience', 'Cost', 'Vegetarian'))
fig2.tight_layout()
fig2.savefig("ML2_HW3_PeoplePreferencesDistribution.png")
#
# From this output it looks like most of the people in our data are pretty opinionated and really care about a single preference.
# This is good news! Mike is the only person that stands out as being even close to having an even distribution of preferences.

############################################################################################################################################
# 9. Think of two metrics to compute the disatistifaction with the group.
############################################################################################################################################
# The distribution of the raw computed scores comes to mind here as well. We saw that converting people's ranks imposed a false sense of importance
# for those scores that may be close to one another. Final scores that were close to one another also may tell us that the winning restaurant
# may have only won by a small margin - meaning a large number of people will be dissatisfied (notably the second losing restaurant).
fig3, ax3 = plt.subplots(1,1, figsize=(5,5))
ax3.hist(myRestaurantRater.getRestaurantRawScoreTotals())
ax3.set(xlabel="Restaurant Raw Score", ylabel="Number of Restaurants", title="Distribution of Restaurant Scores")
fig3.tight_layout()
fig3.savefig("ML2_HW3_RestaurantRawScoreDistribution.png")
#
# This histogram does demonstrate that there are 3 restaurants with very similar scores (between 35 and 37), which means those people in the
# second and third losing restaurants were a good fit for those restaurants (restaurant score scaling issues aside!)

# K-means clustering also comes to mind. While the PCA plot of PC1 against PC2 for the people's preferences clearly demonstrate that there
# should be 3 groups, we could go a step further and quantify the clusters (perhaps using silhouette analysis).

############################################################################################################################################
# 10. Should you split in two groups today?
############################################################################################################################################
pca_people = PCA(n_components=2)
people_matrix_PCA = pca_people.fit_transform(myRestaurantRater.getPeopleMatrix())
print("PCA People Matrix 2 Principal Components")
print(pca_people.components_)
print("PCA People Matrix explained variance:")
print(pca_people.explained_variance_ratio_)

fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.scatter(people_matrix_PCA[:,0], people_matrix_PCA[:,1])
ax.set(xlabel="PC 1", ylabel="PC 2", title="PCA - People Matrix")
fig.tight_layout()
fig.savefig("ML2_HW3_PeopleMatrixPCA.png")
#
# Nearly 80% of the variance in the people matrix is explained by PC1 and PC2. Plotting the output suggests that the group should be split into 3.

############################################################################################################################################
# 11. Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?
############################################################################################################################################
# Easy. Drop all cheaper restaurants, because the big man is paying!
onlyexpensiverestaurants = {}
for key in restaurants.keys():
    if restaurants[key]['cost'] <= 2:
        onlyexpensiverestaurants[key] = restaurants[key]

myRestaurantRaterExpensive = RestaurantRater(people, onlyexpensiverestaurants) # instantiate new object with new expensive restaurants

print("The expensive restaurant order is:")
print(myRestaurantRaterExpensive.getRestaurants())
print("The raw sum of each column (restaurant) is as follows:")
print(myRestaurantRaterExpensive.getRestaurantRawScoreTotals())
print("The rank of each restaurant descending is as follows:")
print(myRestaurantRaterExpensive.getRankedDesc(myRestaurantRaterExpensive.getRestaurantRankScoreTotals()))

############################################################################################################################################
# 12. Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants. Can you
# find their weight matrix?
############################################################################################################################################
#
# I couldn't really think of an easy way to do this, so I'm going to muscle through it with brute force!
#
# This seems to run pretty well! The obvious issue is that we're evaluating ranks, which means we will only ever get close to their
# actual preferences - or only very accurate by chance. This magnifies the problems identified earlier in this homework around converting to ranks.
#
# If we wanted to get even crazier with this we could run this and flag multiple sets of preferences that produced the correct rankings to understand
# the sensitivity and range of the preferences even more.

def find_weights_from_optimal_ordering(iterations, optimalOrder):
    # convert optimalOrder to list for easy list comparison later
    optimalOrder = list(optimalOrder)
    # loop for iterations.....
    for _ in range(iterations):
        # generate a weighting set
        weights = np.random.dirichlet(np.ones(4),size=1)[0]
        # produce raw linear combinations against restaurant matrix
        weightsDotRestaurantRaw = np.matmul(weights, myRestaurantRater.getRestaurantsMatrix().T)
        # convert the raw linear combinations to descending ranks
        seq = sorted(weightsDotRestaurantRaw)
        index_asc = [seq.index(v) for v in weightsDotRestaurantRaw]
        index_inv = np.absolute(np.array(index_asc) - len(list(index_asc)))
        seq2 = sorted(index_inv)
        index_desc = [seq2.index(v) for v in index_inv]
        ranks_list = list(np.array(index_desc) + 1)
        # compare the generated ranks to the provided ordering
        if ranks_list == optimalOrder:
            print("Person Weights Found!")
            print("Willingness to Travel: {}".format(weights[0]))
            print("Desire New Experience: {}".format(weights[1]))
            print("Cost: {}".format(weights[2]))
            print("Vegetarian: {}".format(weights[3]))
            break

print("#########################################################################################")
print("Skip\'s optimal order in descending rank is: {0}".format(myRestaurantRater.getPeopleDotRestaurantsRankDesc()[5]))
find_weights_from_optimal_ordering(50000, myRestaurantRater.getPeopleDotRestaurantsRankDesc()[5])
print("Skip\'s actual weights: {0}".format(people["Skip"]))

print("#########################################################################################")
print("Moe\'s optimal order in descending rank is: {0}".format(myRestaurantRater.getPeopleDotRestaurantsRankDesc()[7]))
find_weights_from_optimal_ordering(50000, myRestaurantRater.getPeopleDotRestaurantsRankDesc()[7])
print("Moe\'s actual weights: {0}".format(people["Moe"]))

print("#########################################################################################")
print("Bob\'s optimal order in descending rank is: {0}".format(myRestaurantRater.getPeopleDotRestaurantsRankDesc()[1]))
find_weights_from_optimal_ordering(50000, myRestaurantRater.getPeopleDotRestaurantsRankDesc()[1])
print("Bob\'s actual weights: {0}".format(people["Bob"]))
