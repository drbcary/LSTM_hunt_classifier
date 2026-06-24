import pandas as pd
import os
import datetime
import math

def frames_to_hms(frames, fps):
    """
    Converts a frame count to a time string format HH:MM:SS.
    """
    if pd.isna(frames) or frames == "":
        return ""
    
    total_seconds = int(frames) / fps
    
    # Use datetime to format easily
    return str(datetime.timedelta(seconds=int(total_seconds)))

def analyze_hunt_data(source_directory, fps=30):
    """
    Iterates through .xlsx files in the source_directory, processes them,
    and creates a new _analyzed.xlsx file for each.
    """
    
    # extensive list of columns for the output dataframe
    output_columns = [
        'Cricket_Number', 
        'Enter_Frame', 
        'Enter_Time', 
        'Capture_Frame', 
        'Capture_Time', 
        'Time_to_Capture_Frames', 
        'Time_to_Capture', 
        'Pursuit_Time_Frames', 
        'Pursuit_Time', 
        'Latency_to_Attack_Frames', 
        'Latency_to_Attack', 
        'Remarks', 
        'Include?'
    ]

    print(f"Processing files in: {source_directory}")

    for filename in os.listdir(source_directory):
        # Process only .xlsx files and skip files that are already analyzed
        if filename.endswith(".xlsx") and "_analyzed" not in filename:
            file_path = os.path.join(source_directory, filename)
            print(f"Analyzing: {filename}...")

            try:
                # Load the raw data
                df = pd.read_excel(file_path)
                
                # Standardize column names to avoid case sensitivity issues
                df.columns = df.columns.str.strip()
                
                # List to hold processed rows
                analyzed_data = []
                
                # Temporary dictionary to hold the current hunt's data
                current_hunt = {}

                for index, row in df.iterrows():
                    event = str(row['Event']).lower().strip()
                    
                    if event == 'cricket enter':
                        # If we were already tracking a hunt, save it before starting the new one
                        if current_hunt:
                            analyzed_data.append(process_hunt_metrics(current_hunt, fps))
                        
                        # Start a new hunt record
                        current_hunt = {
                            'Cricket_Number': row['Cricket number'],
                            'Enter_Frame': row['Frame'],
                            'Pursuit_Start_Frame': None,
                            'Capture_Frame': None,
                            'Remarks': row['Remarks'] if pd.notna(row['Remarks']) else ""
                        }

                    elif event == 'pursuit start':
                        # Only set the start frame if it hasn't been set yet (captures the FIRST occurrence)
                        if current_hunt and current_hunt.get('Pursuit_Start_Frame') is None:
                            current_hunt['Pursuit_Start_Frame'] = row['Frame']

                    elif event == 'capture':
                        if current_hunt:
                            current_hunt['Capture_Frame'] = row['Frame']

                # Append the final hunt if it exists
                if current_hunt:
                    analyzed_data.append(process_hunt_metrics(current_hunt, fps))

                # Create the final DataFrame
                result_df = pd.DataFrame(analyzed_data, columns=output_columns)

                # Generate output filename
                output_filename = filename.replace(".xlsx", "_analyzed.xlsx")
                output_path = os.path.join(source_directory, output_filename)

                # Save to Excel
                result_df.to_excel(output_path, index=False)
                print(f"Saved: {output_filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

def process_hunt_metrics(hunt_data, fps):
    """
    Calculates the specific diffs and times for a single hunt dictionary.
    """
    enter_frame = hunt_data.get('Enter_Frame')
    pursuit_frame = hunt_data.get('Pursuit_Start_Frame')
    capture_frame = hunt_data.get('Capture_Frame')
    remarks = hunt_data.get('Remarks', "")

    # Initialize calculations as None/Empty
    time_to_capture_frames = None
    pursuit_time_frames = None
    latency_frames = None

    # Calculate Frames Diffs if data exists
    if pd.notna(enter_frame) and pd.notna(capture_frame):
        time_to_capture_frames = int(capture_frame - enter_frame)

    if pd.notna(pursuit_frame) and pd.notna(capture_frame):
        pursuit_time_frames = int(capture_frame - pursuit_frame)

    if pd.notna(enter_frame) and pd.notna(pursuit_frame):
        latency_frames = int(pursuit_frame - enter_frame)

    # Determine Inclusion logic
    # If remarks exist, Exclude. Otherwise, Include.
    include_status = "Exclude" if remarks else "Include"

    return {
        'Cricket_Number': hunt_data.get('Cricket_Number'),
        'Enter_Frame': enter_frame,
        'Enter_Time': frames_to_hms(enter_frame, fps),
        'Capture_Frame': capture_frame,
        'Capture_Time': frames_to_hms(capture_frame, fps),
        'Time_to_Capture_Frames': time_to_capture_frames,
        'Time_to_Capture': frames_to_hms(time_to_capture_frames, fps) if time_to_capture_frames is not None else "",
        'Pursuit_Time_Frames': pursuit_time_frames,
        'Pursuit_Time': frames_to_hms(pursuit_time_frames, fps) if pursuit_time_frames is not None else "",
        'Latency_to_Attack_Frames': latency_frames,
        'Latency_to_Attack': frames_to_hms(latency_frames, fps) if latency_frames is not None else "",
        'Remarks': remarks,
        'Include?': include_status
    }

# --- Configuration ---
if __name__ == "__main__":
    # Change this path to your actual folder path containing the Excel files
    # Use raw string (r"path") or forward slashes for Windows paths
    TARGET_DIRECTORY = r"Z:\LucasWahl\Exp 6 -  Scn2a mut naive hunt\Scored excel files\test_analyzed" 
    
    # Set Video FPS
    VIDEO_FPS = 30

    analyze_hunt_data(TARGET_DIRECTORY, VIDEO_FPS)