import streamlit as st
from helpers import create_docs

def main():
    st.set_page_config(page_title="Bill Extractor")
    st.title("Bill Extractor AI Assistant")

    # Upload Bills
    pdf_files = st.file_uploader(
        "Upload your bills in PDF format only",
        type=['pdf'],
        accept_multiple_files=True
    )

    extract_button = st.button("Extract bill data...")

    if extract_button:
        with st.spinner("Extracting..."):
            data_frame = create_docs(pdf_files)
            st.write(data_frame.head())
            #data_frame["AMOUNT"] = data_frame["AMOUNT"].astype(float)
            #st.write("Average bill amount: ", data_frame["AMOUNT"].mean())

            # Convert to CSV
            convert_to_csv = data_frame.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download as CSV",
                convert_to_csv,
                "CSV_Bills.csv",
                "text/csv",
                key="download-csv"
            )

        st.success("Success")



if __name__ == '__main__':
    main()