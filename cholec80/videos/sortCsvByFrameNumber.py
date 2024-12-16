import pandas as pd
import re

def sort_csv_by_surgery_and_frame(input_file, output_file, frame_column, surgery_column):
    """
    Sorts rows in a CSV file first by Surgery_num and then by the numeric part of a specified frame column.
    Includes the frame number in the output.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the sorted CSV file.
        frame_column (str): Name of the column containing the Video_xx_xxxxx format.
        surgery_column (str): Name of the column containing the surgery identifier.
    """
    # Load the CSV file with the semicolon delimiter
    data = pd.read_csv(input_file, sep=';')

    # Extract the numeric part of the frame column
    data['FrameNumber'] = data[frame_column].apply(
        lambda x: int(re.search(r'_(\d+)\.', x).group(1)) if isinstance(x, str) else float('inf')
    )

    # Sort the data by Surgery_num and FrameNumber
    data_sorted = data.sort_values(by=[surgery_column, 'FrameNumber'])

    # Save the sorted data, including the FrameNumber column, to a new CSV file
    data_sorted.to_csv(output_file, index=False, sep=';')

# Example usage:
# Replace 'FrameName' and 'Surgery_num' with the actual column names from your file
sort_csv_by_surgery_and_frame(
    "Cholec80/videos/ROI_Labels.csv",
    "output_sorted2.csv",
    "FrameName",
    "Surgery_num"
)
