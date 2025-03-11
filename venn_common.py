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
import datetime
import re  # For extracting years
from upsetplot import UpSet, from_memberships

# Disable the deprecation warning globally
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set the threshold at the top of the code (you can change this value easily here)
THRESHOLD = 95  # Set your default threshold here

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
    current_year = ""

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
                                # Check if the text is a year (check for 4-digit number)
                                year_match = re.match(r"\b(19|20)\d{2}\b", span['text'])
                                if year_match:
                                    current_year = year_match.group(0)
                                    if current_title.strip():
                                        blue_texts.add((clean_title(current_title.strip()), current_year))
                                        current_title = ""  # Reset current title after saving
                                        current_year = ""  # Reset year after saving
        if current_title.strip():
            blue_texts.add((clean_title(current_title.strip()), current_year))  # Normalize last title if necessary
    finally:
        pdf_document.close()

    return blue_texts

# Fuzzy string matching to find similar titles between two sets
def find_similar_titles(set1, set2, threshold=THRESHOLD):  # Use global threshold variable
    matches = set()
    for title1 in set1:
        for title2 in set2:
            if fuzz.ratio(title1, title2) >= threshold:  # Use threshold variable here
                matches.add(title1)  # Store only one version
    return matches

# Add a button to run the simulation
st.title("Google Scholar PubMatch")

# Add the attribution text below the title
st.markdown(
    '"This common publications matching tool is implemented based on the work done by Rohit Ramachandran, Rutgers University, NJ, USA"'
)

# Add instructions banner
st.markdown("""
### Instructions:
1. Upload at least 2 Google Scholar PDFs. Ensure all the papers are printed to PDF. See 'show more' at the end of the google scholar link.
2. The algorithm will check for combinations of common publications across 2 to N researchers, where N ~=8 before memory limitations hit. N=8 has 247 combinations.
3. Once the analysis is complete, download the comparison results as a CSV file.
4. You can also view the table, Upset plot and Venn diagram showing the overlap of publications between researchers. 
5. Tables and Upset plots are accurate. Venn diagrams for 4 researchers may be a tad inaccurate, but they look nice. 
6. Venn diagrams are absent for 5 or more researchers. 
""")

# Get the current year dynamically
current_year = datetime.datetime.now().year

# Dropdown for year filtering
year_options = ["All Years"] + list(range(current_year, 1950, -1))  # Start from current year
selected_years = st.multiselect("Select years to include:", year_options, default=["All Years"])  # Default: All Years


# If "All Years" is selected, include all years
if "All Years" in selected_years:
    selected_years = set(map(str, range(current_year, 1950, -1)))  # Convert all years to string
else:
    selected_years = set(map(str, [year for year in selected_years if year != "All Years"]))  # Convert selected years to string

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
                extracted_titles_years = extract_blue_text_from_pdf(file)
                # Filter titles by the selected years
                filtered_titles = {title for title, year in extracted_titles_years if year in selected_years}
                researcher_data[researcher_name] = filtered_titles
            else:
                st.warning(f"Skipping {file.name}: Not detected as a Google Scholar PDF.")

        if len(researcher_data) > 1:
            st.subheader("Common Publications Between Researchers")

            all_sets = {name: set(titles) for name, titles in researcher_data.items()}
            comparisons = []

            # Generate all subset combinations (from pairs to all researchers)
            for r in range(2, len(all_sets) + 1):  
                for comb in combinations(all_sets.keys(), r):
                    # Handle combinations dynamically for any number of researchers
                    common_titles = all_sets[comb[0]]
                    for researcher in comb[1:]:
                        common_titles = find_similar_titles(common_titles, all_sets[researcher])  # Fuzzy matching
                    comparisons.append({
                        "Combinations": " ↔ ".join(comb),  
                        "No. of common publications": len(common_titles),
                        "Common Publications": ", ".join(common_titles),  # Add the common titles for CSV export
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
            
            
            # Generate the UpSet plot from the comparisons data dynamically
            st.subheader("UpSet Plot of Common Publications")
            researcher_names = list(researcher_data.keys())
            upset_data = {}

            for _, row in df_comparisons.iterrows():
                combination = row["Combinations"]
                common_publications_count = row["No. of common publications"]
                researchers_in_comb = combination.split(" ↔ ")
                membership = tuple(researcher in researchers_in_comb for researcher in researcher_names)
                if membership in upset_data:
                    upset_data[membership] += common_publications_count
                else:
                    upset_data[membership] = common_publications_count

            index = pd.MultiIndex.from_tuples(upset_data.keys(), names=researcher_names)
            data_counts = pd.Series(list(upset_data.values()), index=index)

            upset = UpSet(data_counts, subset_size="auto", show_counts="%d", sort_by="cardinality", facecolor="blue")
            fig = plt.figure(figsize=(12, 8))
            upset.plot(fig=fig)
            st.pyplot(fig)
                            
           
                        
            
            

            # Only plot the Venn diagram if there are 2, 3, or 4 researchers

            if len(researcher_data) > 1:
                # Skip the Venn diagram if more than 4 researchers are selected
                if len(researcher_data) > 4:
                    st.warning("Venn diagrams are only available for up to 4 researchers. Skipping Venn diagram.")
                else:
                    # Prepare the set names and total publications per researcher
                    researcher_names = list(all_sets.keys())
                    total_publications = {name: len(all_sets[name]) for name in researcher_names}

                    # Initialize dictionaries to store intersection counts
                    intersection_counts = {}
                    for r in range(1, len(researcher_names) + 1):
                        for comb in combinations(researcher_names, r):
                            intersection_counts[comb] = 0

                    # Populate intersection counts from df_comparisons
                    for _, row in df_comparisons.iterrows():
                        researchers_in_comb = tuple(row["Combinations"].split(" ↔ "))
                        count = row["No. of common publications"]
                        intersection_counts[researchers_in_comb] = count

                    # Compute singleton counts (e.g., publications unique to each researcher)
                    singleton_counts = {}
                    for name in researcher_names:
                        total = total_publications[name]
                        # Subtract all intersections involving this researcher
                        for comb, count in intersection_counts.items():
                            if name in comb:
                                total -= count
                        singleton_counts[name] = max(total, 0)  # Ensure non-negative

                    if len(researcher_names) == 2:
                        name1, name2 = researcher_names
                        venn_data = [
                            singleton_counts[name1],  # Only name1
                            singleton_counts[name2],  # Only name2
                            intersection_counts[(name1, name2)]  # name1 & name2
                        ]
                        fig, ax = plt.subplots(figsize=(6, 6))
                        v = venn.venn2(
                            subsets=venn_data,
                            set_labels=researcher_names
                        )
                        st.pyplot(fig)

                    elif len(researcher_names) == 3:
                        name1, name2, name3 = researcher_names
                        venn_data = [
                            singleton_counts[name1],  # Only name1
                            singleton_counts[name2],  # Only name2
                            intersection_counts[(name1, name2)],  # name1 & name2
                            singleton_counts[name3],  # Only name3
                            intersection_counts[(name1, name3)],  # name1 & name3
                            intersection_counts[(name2, name3)],  # name2 & name3
                            intersection_counts[(name1, name2, name3)]  # name1 & name2 & name3
                        ]
                        fig, ax = plt.subplots(figsize=(6, 6))
                        v = venn.venn3(
                            subsets=venn_data,
                            set_labels=researcher_names
                        )
                        st.pyplot(fig)

                    elif len(researcher_names) == 4:
                        # For venny4py, we need to compute all possible intersections
                        name1, name2, name3, name4 = researcher_names
                        # Create a dictionary of sets representing each intersection
                        venn_sets = {
                            name: set() for name in researcher_names
                        }
                        # Populate the sets with dummy elements based on counts
                        # Singletons
                        for name in researcher_names:
                            count = singleton_counts[name]
                            venn_sets[name].update([f"{name}_unique_{i}" for i in range(count)])
                        
                        # Pairwise intersections
                        for (n1, n2) in combinations(researcher_names, 2):
                            count = intersection_counts[(n1, n2)]
                            titles = [f"{n1}_{n2}_{i}" for i in range(count)]
                            venn_sets[n1].update(titles)
                            venn_sets[n2].update(titles)
                        
                        # Triple intersections
                        for (n1, n2, n3) in combinations(researcher_names, 3):
                            count = intersection_counts.get((n1, n2, n3), 0)
                            titles = [f"{n1}_{n2}_{n3}_{i}" for i in range(count)]
                            venn_sets[n1].update(titles)
                            venn_sets[n2].update(titles)
                            venn_sets[n3].update(titles)
                        
                        # Quadruple intersection
                        count = intersection_counts.get((name1, name2, name3, name4), 0)
                        titles = [f"{name1}_{name2}_{name3}_{name4}_{i}" for i in range(count)]
                        for name in researcher_names:
                            venn_sets[name].update(titles)

                        # Generate the 4-set Venn diagram with venny4py
                        fig = venny4py(sets=venn_sets)
                        st.pyplot(fig)

# ... (rest of your code: execution time) ...
# ... (rest of your code: execution time) ...
        # elif len(publication_sets) == 4:
        #     # Create modified sets accounting for fuzzy matching
        #     fuzzy_sets = {}
        
        #     for i in range(4):
        #         current_set = publication_sets[i].copy()
        #         for j in range(4):
        #             if i != j:
        #                 similar_titles = find_similar_titles(publication_sets[i], publication_sets[j], threshold=THRESHOLD)
        #                 current_set.update(similar_titles)
        #         fuzzy_sets[researcher_names[i]] = current_set
        
        #     # Now plot the Venn diagram using fuzzy-enhanced sets
        #     fig = venny4py(sets=fuzzy_sets)
        #     st.pyplot(fig)


        # Record end time and calculate execution time
        execution_time = (time.time() - start_time)/60
        st.markdown(f"<h3 style='font-size: 30px;'>Simulation time: {execution_time:.2f} mins</h3>", unsafe_allow_html=True)


    else:
        st.warning("Please upload at least two Google Scholar PDFs to run the simulation.")
