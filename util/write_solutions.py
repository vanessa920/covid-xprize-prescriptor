# Write solutions for one geoid query to csv

# @author Andrew Zhou

import pandas as pd

NPI_COLUMNS = ['C1_School closing',
               'C2_Workplace closing',
               'C3_Cancel public events',
               'C4_Restrictions on gatherings',
               'C5_Close public transport',
               'C6_Stay at home requirements',
               'C7_Restrictions on internal movement',
               'C8_International travel controls',
               'H1_Public information campaigns',
               'H2_Testing policy',
               'H3_Contact tracing',
               'H6_Facial Coverings']

def write_solutions(prescriptions, country_name, region_name, start_date, num_days, output_csv):
 
    for i in range(len(prescriptions)):
        prescription_np = prescriptions[i].numpy()
        df = pd.DataFrame(prescription_np, columns=NPI_COLUMNS)
        df[NPI_COLUMNS] = df[NPI_COLUMNS].astype('int32')
        df["Date"] = pd.date_range(start_date, periods=num_days, freq='D')
        df["CountryName"] = country_name
        df["RegionName"] = region_name
        df["RegionName"] = df["RegionName"].fillna(value='')
        df["PrescriptionIndex"] = i
        
        df.to_csv(output_csv, mode="a", header=False, index=False)