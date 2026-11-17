import streamlit as st


def show_home():

    st.markdown(
        """
        <div class='main-title'>
            🛡️ CYBERSHIELD AI
        </div>

        <div class='sub-title'>
            Enterprise Phishing Detection Platform
        </div>
        """,
        unsafe_allow_html=True
    )

    # ------------------------
    # HERO SECTION
    # ------------------------

    st.markdown(
        """
        <div class='cyber-card'>
            <h2>🚀 Welcome to CyberShield AI</h2>
            <p>
            Detect phishing websites using Machine Learning,
            DNS Intelligence, SSL Validation, and Threat Analytics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ------------------------
    # DASHBOARD METRICS
    # ------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class='metric-card'>
                <div class='metric-title'>
                    URLs Scanned
                </div>

                <div class='metric-value'>
                    11,430
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class='metric-card'>
                <div class='metric-title'>
                    Accuracy
                </div>

                <div class='metric-value'>
                    97.8%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class='metric-card'>
                <div class='metric-title'>
                    Precision
                </div>

                <div class='metric-value'>
                    96.9%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class='metric-card'>
                <div class='metric-title'>
                    F1 Score
                </div>

                <div class='metric-value'>
                    97.1%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    st.write("")

    # ------------------------
    # FEATURES SECTION
    # ------------------------

    st.subheader("🔥 Platform Features")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class='cyber-card'>
            <h4>🔍 Single URL Detection</h4>

            Analyze websites instantly using
            Machine Learning and URL intelligence.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown(
            """
            <div class='cyber-card'>
            <h4>📂 Bulk CSV Scanner</h4>

            Upload thousands of URLs and detect
            phishing websites automatically.
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class='cyber-card'>
            <h4>📊 Analytics Dashboard</h4>

            Accuracy, Precision, Recall,
            F1 Score, and Confusion Matrix.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown(
            """
            <div class='cyber-card'>
            <h4>⬇ Download Reports</h4>

            Export phishing detection results
            as CSV reports.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    st.write("")

    # ------------------------
    # ABOUT SECTION
    # ------------------------

    st.subheader("ℹ About")

    st.markdown(
        """
        CyberShield AI is a machine learning-based phishing
        detection platform designed to identify malicious
        URLs and improve cybersecurity awareness.

        Features include:

        - URL Analysis
        - Bulk Detection
        - Threat Analytics
        - CSV Report Export
        - DNS & SSL Validation
        """
    )