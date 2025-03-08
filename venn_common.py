import streamlit as st
import fitz  
from io import BytesIO
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from venny4py.venny4py import venny4py  # Import venny4py for 4-set Venn diagrams
import matplotlib_venn as venn  # Correct import for venn library
from fuzzywuzzy import fuzz  # Import fuzzywuzzy for fuzzy string matching
import time  # Import time module to track execution time

# Disable the deprecation warning globally
st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to normalize publication titles
def clean_title(title):
    return " ".join(title.lower().strip().replace("-", " ").split())  # Normalize hyphens, spaces, and case

# Strip the .pdf extension from file name
def clean_filename(file_name):
    return file_name.rsplit('.', 1)[0]  # Remove the .pdf extension

def is_google_scholar(file_object):
    if file_object is None or file_object.getbuffer().nbytes == 0:
        file_name = getattr(file_object, "name", "Unknown file")
        st.error(f"The uploaded file '{file_name}' is empty or invalid.")
        return False

    file_object.seek(0)
    pdf_document = fitz.open(stream=file_object.read(), filetype="pdf")
    page1 = pdf_document.load_page(0)

    topmost_y = float('inf')
    topmost_text = ""

    for block in page1.get_text("dict")["blocks"]:
        block_type = block.get("type", 0)
        if block_type == 0: 
            block_bbox = block.get("bbox", [])
            if block_bbox and block_bbox[1] < topmost_y:
                topmost_y = block_bbox[1]
                topmost_text = " ".join(
                    span['text'] for line in block.get("lines", [])
                    for span in line.get("spans", [])
                )

    pdf_document.close()
    return "Google Scholar" in topmost_text

def is_blue(color_int):
    r = (color_int >> 16) & 0xFF
    g = (color_int >> 8) & 0xFF
    b = color_int & 0xFF
    return b > r and b > g

def extract_blue_text_from_pdf(file_object):
    blue_texts = set()
    current_title = ""

    if file_object is None or file_object.getbuffer().nbytes == 0:
        return set()

    file_object.seek(0)
    pdf_document = fitz.open(stream=file_object.read(), filetype="pdf")

    try:
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            for text_instance in page.get_text("dict")["blocks"]:
                if "lines" in text_instance:
                    for line in text_instance["lines"]:
                        for span in line["spans"]:
                            if is_blue(span['color']):
                                current_title += span['text'] + " "
                            else:
                                if current_title.strip():
                                    blue_texts.add(clean_title(current_title.strip()))  # Normalize title before adding
                                    current_title = ""

        if current_title.strip():
            blue_texts.add(clean_title(current_title.strip()))  # Normalize last title if necessary

    finally:
        pdf_document.close()

    return blue_texts

# Fuzzy string matching to find similar titles between two sets
def find_similar_titles(set1, set2, threshold=80):
    matches = set()
    for title1 in set1:
        for title2 in set2:
            if fuzz.ratio(title1, title2) >= threshold:  # Allowing 80% similarity
                matches.add(title1)  # Store only one version
    return matches

# Add a button to run the simulation
st.title("Google Scholar Common Publications Checker")

# Add the attribution text below the title
st.markdown(
    '"This common publications checker is implemented based on the work done by Rohit Ramachandran, Rutgers University, NJ, USA"'
)

# Add instructions banner
st.markdown("""
### Instructions:
1. Upload at least 2 Google Scholar PDFs. Ensure all the papers are printed to PDF. See 'show more' bottom of google scholar link.
2. The system will check for common publications across these researchers.
3. Once the analysis is complete, download the comparison results as a CSV file.
4. You can also view the table and Venn diagram showing the overlap of publications between researchers. 
""")

uploaded_files = st.file_uploader(
    "Upload Google Scholar PDFs of researchers",
    type=["pdf"],
    accept_multiple_files=True
)

# Add a button for running the simulation
run_simulation_button = st.button("Run Simulation")

# If the button is pressed, start the simulation
if run_simulation_button:
    # Record start time
    start_time = time.time()

    if uploaded_files:
        researcher_data = {}

        for file in uploaded_files:
            if is_google_scholar(file):
                researcher_name = clean_filename(file.name)  # Clean the filename by removing .pdf
                extracted_titles = extract_blue_text_from_pdf(file)
                researcher_data[researcher_name] = extracted_titles
            else:
                st.warning(f"Skipping {file.name}: Not detected as a Google Scholar PDF.")

        if len(researcher_data) > 1:
            st.subheader("Common Publications Between Researchers")

            all_sets = {name: set(titles) for name, titles in researcher_data.items()}
            comparisons = []

            # Generate all subset combinations (from pairs to all researchers)
            for r in range(2, len(all_sets) + 1):  
                for comb in combinations(all_sets.keys(), r):
                    if len(comb) == 2:
                        # Use fuzzy matching (instead of exact match) to find common titles for pairs
                        common_titles = find_similar_titles(all_sets[comb[0]], all_sets[comb[1]], threshold=80)
                        comparisons.append({
                            "Combinations": " ↔ ".join(comb),  
                            "No. of common publications": len(common_titles),
                            # "Common Publications" not included in the table display
                            "Common Publications": ", ".join(common_titles),  # Add the common titles for CSV export
                        })
                    elif len(comb) == 3:
                        # Handle the case for 3 researchers by calculating the intersection of all three sets
                        common_titles_1_2 = find_similar_titles(all_sets[comb[0]], all_sets[comb[1]], threshold=80)
                        common_titles_2_3 = find_similar_titles(all_sets[comb[1]], all_sets[comb[2]], threshold=80)
                        common_titles_1_3 = find_similar_titles(all_sets[comb[0]], all_sets[comb[2]], threshold=80)
                        common_titles_all = find_similar_titles(common_titles_1_2, common_titles_2_3, threshold=80)
                        common_titles_all = find_similar_titles(common_titles_all, common_titles_1_3, threshold=80)
                        comparisons.append({
                            "Combinations": " ↔ ".join(comb),  
                            "No. of common publications": len(common_titles_all),
                            # "Common Publications" not included in the table display
                            "Common Publications": ", ".join(common_titles_all),  # Add the common titles for CSV export
                        })

            # Create the DataFrame with the common publications count
            df_comparisons = pd.DataFrame(comparisons)

            # Reset the index and start from 1
            df_comparisons.index = range(1, len(df_comparisons) + 1)

            # Display the table without "Common Publications"
            st.dataframe(
                df_comparisons.drop(columns=["Common Publications"]).style.set_properties(**{
                    'text-align': 'left',
                    'white-space': 'pre-wrap'
                }),
                height=(50 + len(df_comparisons) * 35),  
                width=800
            )

            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            # Provide option to download the table as CSV
            csv_data = convert_df(df_comparisons)
            st.download_button(
                label="Download Comparison as CSV",
                data=csv_data,
                file_name="publication_comparisons.csv",
                mime="text/csv"
            )

        # Only plot the Venn diagram if there are 2, 3, or 4 researchers
        if len(researcher_data) > 1:  # Removed the check for > 4 researchers
            # Prepare the set names and publication sets
            researcher_names = list(all_sets.keys())
            publication_sets = list(all_sets.values())

            if len(publication_sets) == 2:
                common_titles = find_similar_titles(publication_sets[0], publication_sets[1], threshold=80)
                venn_data = [len(publication_sets[0] - common_titles), len(publication_sets[1] - common_titles), len(common_titles)]
                fig, ax = plt.subplots(figsize=(6, 6))
                v = venn.venn2(
                    subsets=venn_data,
                    set_labels=researcher_names
                )
                st.pyplot(fig)  # Explicitly passing the figure object

            elif len(publication_sets) == 3:
                common_titles_1_2 = find_similar_titles(publication_sets[0], publication_sets[1], threshold=80)
                common_titles_2_3 = find_similar_titles(publication_sets[1], publication_sets[2], threshold=80)
                common_titles_1_3 = find_similar_titles(publication_sets[0], publication_sets[2], threshold=80)
                common_titles_all = find_similar_titles(common_titles_1_2, common_titles_2_3, threshold=80)
                common_titles_all = find_similar_titles(common_titles_all, common_titles_1_3, threshold=80)
                venn_data = [
                    len(publication_sets[0] - common_titles_1_2 - common_titles_1_3),
                    len(publication_sets[1] - common_titles_1_2 - common_titles_2_3),
                    len(common_titles_1_2),
                    len(publication_sets[2] - common_titles_2_3 - common_titles_1_3),
                    len(common_titles_1_3),
                    len(common_titles_2_3),
                    len(common_titles_all)
                ]
                fig, ax = plt.subplots(figsize=(6, 6))  # Ensure we pass the figure object
                v = venn.venn3(
                    subsets=venn_data,
                    set_labels=researcher_names
                )
                st.pyplot(fig)  # Explicitly passing the figure object

            elif len(publication_sets) == 4:
                # Handle 4 sets using venny4py
                sets = {researcher_names[i]: publication_sets[i] for i in range(4)}  # Map researcher names to sets
                fig = venny4py(sets=sets)  # Generate the 4-set Venn diagram with venny4py
                st.pyplot(fig)  # Explicitly passing the figure object

    # Record end time and compute total simulation time
    end_time = time.time()
    total_time = (end_time - start_time)/60

    # Display the total simulation time
    st.subheader(f"Total simulation time: {total_time:.2f} minutes")
