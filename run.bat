@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting Streamlit app...
streamlit run Exsrt.py
pause
