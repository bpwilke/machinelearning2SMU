import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # for centering numeric features for modeling
from sklearn.preprocessing import OneHotEncoder # hot! sizle.
from sklearn.linear_model import LogisticRegression # for modeling
from sklearn.model_selection import cross_validate # k-fold for gauging model accuracy
from collections import OrderedDict # for sorting a dict
from operator import itemgetter # hack needed later to sort coeffecients

# load the headers for indexing later
headers = np.loadtxt("claim.sample.csv", delimiter=",", dtype="U30")
headers = headers[0,]
headers = np.char.strip(headers, chars="\"") # strip extra quotes from string
#print(headers)

##########################################################################################################################################################################
#1. J-codes are procedure codes that start with the letter 'J'.
#A. Find the number of claim lines that have J-codes.
##########################################################################################################################################################################

# load Procedure.Code as string data into numpy array, skip header
procedurecode = np.loadtxt("claim.sample.csv", delimiter=",", usecols=(np.where(headers == "Procedure.Code")[0][0]), skiprows=1, dtype='U30')
procedurecode = np.char.strip(procedurecode, chars="\"") # strip extra quotes from string

# find elements that contain a "J" and sum total count
print("The number of claim lines that have J-Code is: {0}".format(np.sum(np.core.defchararray.startswith(procedurecode,"J"))))

##########################################################################################################################################################################
#B. How much was paid for J-codes to providers for 'in network' claims?
##########################################################################################################################################################################

# find index of all J-codes
procedurecode_idx = np.core.defchararray.startswith(procedurecode,"J")

# load In.Out.Of.Network as string data into numpy array, skip header
inoutofnetwork = np.loadtxt("claim.sample.csv", delimiter=",", usecols=(np.where(headers == "In.Out.Of.Network")[0][0]), skiprows=1, dtype='U3')
inoutofnetwork = np.char.strip(inoutofnetwork, chars="\"") # strip extra quotes from string

# find index of all "I" in-network
inoutofnetwork_idx = np.core.defchararray.startswith(inoutofnetwork,"I")

# find those rows that have both J-code and in-network
jcode_and_innetwork = np.logical_and(procedurecode_idx, inoutofnetwork_idx)

# load Provider.Payment.Amount as float for summation
providerpaymentamt = np.loadtxt("claim.sample.csv", delimiter=",", usecols=(np.where(headers == "Provider.Payment.Amount")[0][0]), skiprows=1)

print("Total Amount paid for All Claims: ${0:.2f}".format(np.sum(providerpaymentamt)))
print("Amount paid only for J-Codes and In-Network Claims: ${0:.2f}".format(np.sum(providerpaymentamt[jcode_and_innetwork]))) # index by j-code + innetwork, sum

##########################################################################################################################################################################
#C. What are the top five J-codes based on the payment to providers?
##########################################################################################################################################################################

# find all j-codes, then find uniques
unique_jcodes = list(set([code for code in procedurecode if code.startswith("J")])) # mmmm pythonic

# initialize and create dictionary to hold our data
jcodes_totals = {}
for jcode in unique_jcodes:
    jcodes_totals.update({jcode : 0})

# populate our dictionary of j-codes
for index, code in enumerate(procedurecode):
    if code in jcodes_totals:  # check to see if code is in J-code list
        jcodes_totals[code] = jcodes_totals[code] + providerpaymentamt[index] # if it is then add the payment associated to the total

# sort our dictionary by provider payment totals
jcodes_totals_sorted = OrderedDict(sorted(jcodes_totals.items(), key=lambda x: x[1], reverse=True)) # reverse=True!! shows in correct descending order

# print all results
print("Top 5 J-Codes Based on Payment to Providers:")
for count, key in enumerate(jcodes_totals_sorted):
    if count < 5:
        print("J-Code {0} Total is: ${1:.2f}".format(key, jcodes_totals_sorted[key]))

##########################################################################################################################################################################
#2. For the following exercises, determine the number of providers that were paid for at least one J-code. Use the J-code claims for these providers to complete the following exercises.
#A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each provider versus the number of paid claims.
##########################################################################################################################################################################

# load Provider.ID as string data into numpy array, skip header
providerIDs = np.loadtxt("claim.sample.csv", delimiter=",", usecols=(np.where(headers == "Provider.ID")[0][0]), skiprows=1, dtype='U25')
providerIDs = np.char.strip(providerIDs, chars="\"") # strip extra quotes from string

# find all provider IDs, then find uniques
unique_providerIDs = list(set([providerID for providerID in providerIDs]))

provider_payment = [] # intialize provider payment list

# checks to see if Provider has at least 1 Jcode procedure code (missed this in requirements at first)
def providerHasAtLeastOneJcode(providerID):
    jcode = False
    for procedure in procedurecode[providerIDs == providerID]: # check each procedure in limited list just for that provider
        if procedure.startswith('J'): # standard python string method
            jcode = True
            break # break because you found one, don't need to look any more
    return jcode

for provider in unique_providerIDs: # for each provider

    if providerHasAtLeastOneJcode(provider): # only process the provider if they have at least 1 J Code
        provider_payment_entry = {} # intialize provider payment entry to add to our list
        non_payment = 0
        paid = 0
        all_payments = providerpaymentamt[providerIDs == provider] # get all payments by provider
        for pay_idx, procedure in enumerate(procedurecode[providerIDs == provider]): # check each procedure by provider
            if procedure.startswith('J'): # is the procedure a J-code?
                if all_payments[pay_idx] == 0:  # if it is J-code, check corresponding payment amount...is it zero?
                    non_payment += 1    # classify each payment as "paid" or "non-payment"
                else:
                    paid += 1

        # calculate percentages for easy use later
        total_payments = paid + non_payment
        pay_pct = (paid / total_payments) * 100
        non_pay_pct = 100 - pay_pct

        # form our entry dict; add our dict entry to our list
        provider_payment_entry.update({"providerID": provider, "payment": paid, "non-payment": non_payment, "pay_pct": pay_pct, "non-pay_pct":  non_pay_pct})
        provider_payment.append(provider_payment_entry)

# function to examine the count of paid and non-paid by provider (takes our payment dict)
def show_total_bar_chart(paymentdict):
    labels = [entry["providerID"] for entry in paymentdict]
    pay = [entry["payment"] for entry in paymentdict]
    nonpay = [entry["non-payment"] for entry in paymentdict]

    width = .35
    ind = np.arange(len(labels))

    fig1 = plt.figure(figsize=(7,7))
    axes1 = fig1.add_subplot()

    axes1.barh(ind, nonpay, width, label="Non-Payment", color="green")
    axes1.barh(ind + width, pay, width, label="Paid", color="magenta")

    plt.yticks(ind, labels)
    plt.legend(loc="best")
    plt.title("Providers Total Paid vs. Total Non-Payment")
    plt.xlabel("Claims Paid and Non-Payment")
    plt.ylabel("Provider ID")

    fig1.tight_layout()
    fig1.savefig("Wilke_ML2_HW2_ProvidersTotalStacked.png")

# process plot
show_total_bar_chart(provider_payment)

# function to take our payment dict and plot a stacked bar chart to examine percentage of split
def show_stacked_bar_chart(paymentdict):
    labels = [entry["providerID"] for entry in paymentdict]
    pay = [entry["pay_pct"] for entry in paymentdict]
    nonpay = [entry["non-pay_pct"] for entry in paymentdict]

    ind = np.arange(len(labels))

    fig1 = plt.figure(figsize=(7,7))
    axes1 = fig1.add_subplot()
    axes1.barh(ind, nonpay, label="Non-Payment")
    axes1.barh(ind, pay, left=nonpay, label="Paid")

    plt.yticks(ind, labels)
    plt.legend(loc="best")
    plt.title("Providers Percentage Paid")
    plt.xlabel("Percentage Split of Payment")
    plt.ylabel("Provider ID")
    fig1.tight_layout() # <---- lifesaver.
    fig1.savefig("Wilke_ML2_HW2_ProvidersPercentageStacked.png")

# process plot
show_stacked_bar_chart(provider_payment)

##########################################################################################################################################################################
#B. What insights can you suggest from the graph?
# It's clear that a very large amount of J-code claims are unpaid (which I'll quantify in the next section). In general - most of the providers with a very large amount
# of total claims have a very small amount of paid claims (less than 5% paid). In comparison, providers with fewer total claims seem to have more paid claims (20-40% paid).
#C. Based on the graph, is the behavior of any of the providers concerning? Explain.
# Following the patterns observed above, it appears that FA0001387001 has a high total claims with a staggering amount of them unpaid (<5% paid). FA0001389001 also
# has a high total number of claims with less than 10% of them being paid. There are a few others with small percentage paid (like FA1000015002 and FA1000016002), but their
# total volume isn't as significant.
##########################################################################################################################################################################
#3. Consider all claim lines with a J-code.
#A. What percentage of J-code claim lines were unpaid?
##########################################################################################################################################################################

# Thanks to David's hints - creating numpy structured array !!!!!!!!!!!!!!!!!!!!!!!!!! ######################

# array of dtypes per "field" from raw data
dat_format = ["U15", "i4", "i4", "i4", "U30", "U15", "U15", "U15", "U15", "U15", "U15", "f4", "U10", "U5",
             "U5", "U5", "U5", "U5", "f8", "f8", "i4", "i4", "i4", "U15", "U4", "U4", "U4", "U22", "U22"]

# create my dtype dictionary
dat_spec = {"names": headers, "formats": dat_format} # create dtype constructor format

dat_list = [] # empty list to hold tuples of raw column data

for dat_index, header in enumerate(dat_spec["names"]): # load each column by data type
    col = np.loadtxt("claim.sample.csv", delimiter=",", usecols=(np.where(headers == header)[0][0]), skiprows=1, dtype=dat_spec["formats"][dat_index])
    print("Loading: {0}".format(header))
    if "U" in dat_spec["formats"][dat_index]: # check if data type is string
        col = np.char.strip(col, chars="\"") # strip extra quotes from string
    dat_list.append(tuple(col)) # append column data to my list

dat = np.zeros(len(dat_list[0]), dtype=dat_spec) # create empty numpy structured array, assigning names and dtype

# populate structured array from my list of tuples (raw columnar data)
for index, header in enumerate(headers):
    dat[header] = dat_list[index]

############## DOH! this is basically doing the same thing, but would need to clean up the strings (remove quotation marks)
dat2 = np.genfromtxt("claim.sample.csv", delimiter=",", dtype=dat_spec, skip_header=1)

unpaid = sum(dat[np.core.defchararray.startswith(dat["Procedure.Code"],"J")]["Provider.Payment.Amount"] == 0)
total = len(dat[np.core.defchararray.startswith(dat["Procedure.Code"],"J")]["Provider.Payment.Amount"])
print("Percentage of J-code claim lines unpaid: {0:.2f}%".format(unpaid / total * 100))

##########################################################################################################################################################################
#B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.
##########################################################################################################################################################################
# I chose to use a fairly basic logistic regression classifier for this data. This choice was made because we are predicting a binary (paid vs. unpaid) target.
# This modeling approach was also chosen as I will be analyzing features/values that contribute the most to the predictive accuracy of the model, which is harder to
# do for quick analysis using ensemble classification tree based methods.
######################### PREP DATA ######################################################################################################################################

# reduce full data set to only those row with J-codes
model_dat = dat[np.core.defchararray.startswith(dat["Procedure.Code"],"J")]

# create binary target (1,0) :: THIS SETS THE TARGET TO 1/TRUE FOR "UNPAID", SINCE THAT'S WHAT WE'RE PREDICTING LATER
target = [1 if each == True else 0 for each in model_dat["Provider.Payment.Amount"] == 0] # look at that if/else comprehension bad boy!! mmm pythonic.

# remove Provider.Payment.Amount as to not include in the model training (it's what we're predicting)
remove_headers = list(headers) # get and convert headers np array to python list
remove_headers.remove("Provider.Payment.Amount") # remove Provider.Payment.Amount from list
model_dat = model_dat[remove_headers] # index model_dat by new headers missing Provider.Payment.Amount

# create lists of numeric data and data that we would like to use from string/discrete factor
numeric_cols = ["Subscriber.Payment.Amount", "Claim.Charge.Amount"]
dummy_cols = ["Provider.ID", "Line.Of.Business.ID", "Service.Code", "In.Out.Of.Network", "Network.ID", "Agreement.ID", "Price.Index", "Claim.Type", "Procedure.Code", "Revenue.Code"]

encode_dat_headers = [] # empty list to hold our one-hot new header names formed from original name + current value

# for each identified string/discrete factor, create one-hot encoded array, horizontal stack each new to create one large array, form and save the new feature names in order
encoder = OneHotEncoder(sparse=False) # instantiate OneHotEncoder()
for loop, col in enumerate(dummy_cols):
    current_dat = model_dat[col]
    current_one_hot = encoder.fit_transform(current_dat.reshape(-1,1))
    if loop == 0:  # this is the first iteration, so initialize one_hot_data with first data
        one_hot_dat = current_one_hot
    else: # if after the first iteration, then horizontal stack the current_one_hot to final one_hot_dat
        one_hot_dat = np.hstack((one_hot_dat, current_one_hot))

    encode_dat_headers.append([col + "-" + name for name in encoder.categories_[0]]) # form the name for each encoded feature; add to list

# list comprehension to "unlist" the new feature names and get one list with each new feature name (since it was a list of lists)
encode_dat_headers = [y for x in encode_dat_headers for y in x]

# for each identified numeric feature, tack on to data array created above, add the feature names in order to the end of the feature list
scaler = StandardScaler() # instantiate a scaler
for col in numeric_cols:
    current_col = scaler.fit_transform(model_dat[col].reshape(-1,1)) # scale current numeric colum
    one_hot_dat = np.hstack((one_hot_dat, current_col))  # tack onto our modeling dataset
    encode_dat_headers.append(col)

##########################################################################################################################################################################
#C. How accurate is your model at predicting unpaid claims?
##########################################################################################################################################################################

lr_clf = LogisticRegression(solver="lbfgs", max_iter=600, random_state=8008)  # instantiate logistic regression classifier
cv_results = cross_validate(lr_clf, one_hot_dat, target, cv=10) # run 10-fold CV on dataset...no hold out. should be enough for this assignment to gauge rough accuracy
#^^ this uses StratifiedKFold as well...which is great, so I don't have to preserve the target distributions myself.
print("10-Fold Logistic Regression Mean Accuracy: {0:.2f}%".format(cv_results["test_score"].mean() * 100))

##########################################################################################################################################################################
#D. What data attributes are predominately influencing the rate of non-payment?
# Of the top 10 coeffecients affecting the predictions of our model Agreement ID shows up 4 times. Agreement ID, likely relates to the contractual stipulations in a health
# insurance plan. There are Agreements that don't cover these J-code treatments and some that do (thus would be excellent predictors of non-payment).
##########################################################################################################################################################################

lr_clf.fit(one_hot_dat, target) # fit the entire data set
one_hot_coefs = list(zip(lr_clf.coef_[0], encode_dat_headers)) # zip together coeffecient with names
one_hot_coefs_abs = [[abs(entry[0]), entry[1]] for entry in one_hot_coefs] # save coeffecient as absolute value
getcoef = itemgetter(0) # needed to sort a list of tuples
one_hot_coefs_abs = sorted(one_hot_coefs_abs, key=getcoef, reverse=True) # store sorted coeffecients by absolute values

# print top 10 feature + attributes by absolute coeffecient
print("The Top 10 Feature/Values by Importance are:")
for count, each in enumerate(one_hot_coefs_abs[0:10]):
        print(str(count + 1) + ". " + str(each[1]))
