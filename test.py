import google.generativeai as genai

genai.configure(api_key="AIzaSyCqxahdXW63NEM4rtkDQ-EhlYHLa3AKKOU")

model = genai.GenerativeModel("gemini-1.5-flash-latest")
response = model.generate_content("Hello Gemini!")
print(response.text)
