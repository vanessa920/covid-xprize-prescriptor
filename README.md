# XPrize Pandemic Response Challenge Prescriptor
Team metis2020's prescriptor for phase II of the XPRIZE Pandemic Response Challenge.


Authored by team member [Andrew Zhou](https://github.com/zhouandrewc).

# Summary

We are tasked to prescribe specific non-pharmaceutical interventions, as defined by  [Oxford](https://www.bsg.ox.ac.uk/research/research-projects/coronavirus-government-response-tracker) to lower the COVID caseload for various countries and regions. However, we also want to minimize the disruptions caused by these interventions. Acknowledging that there is necessarily a tradeoff between these two goals, we want to provide a set of at most 10 prescriptions, each striking a different balance between case reduction and intervention stringency.

# Approach

We construct a Keras model with trainable weights corresponding to the requested NPI (non-pharmaceutical intervention) prescriptions. 

This model takes as input the initial conditions of a given GeoID: the PredictionRatios and NPIs for the 21 days prior to the start date, the population of the GeoID, the total number of cases on the day prior to the start date, and the daily case numbers for the 7 days prior to the start date. It outputs the predicted number of total cases in the prescription period, assuming the trainable weights represent the implemented NPIs for that period.

Beginning with a zero-intervention strategy, we use Keras to calculate the gradients of the NPIs with respect to the caseload, and in so doing find the best NPIs to increment. Repeating this process until the NPIs are maximized gives an array of solutions. We select ten of these solutions, each representing a different stringency-caseload tradeoff. 

# Files

-  [prescribe.py](prescribe.py)

Our main prescription program, which adheres to the API specified by the PRC guidelines. Primarily an interface to the main logic contained in [util](util).

- [util](util)

The main routines to construct and train the model, select the desired prescriptions, and write them to a file. Further details to follow.

- [tasks](tasks)

Example task files from the XPRIZE sandbox.

- [demo.ipynb](example.ipynb)

A demonstration of our code running on an example task: prescribing for the period of February 15th to February 20th, 2021 for all the required GeoIDs.

- [prescriptions/demo_outputs.csv](prescriptions/demo_outputs.csv)

Prescriptions for the demonstration run.

# Future Work

Under construction.