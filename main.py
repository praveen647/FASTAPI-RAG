from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI, HTTPException
import pyrebase
from urllib.parse import urldefrag
import re
from PIL import Image
from io import BytesIO
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel
from typing import List
from fastapi import Depends,status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
import nest_asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
import urllib
import warnings
from pathlib import Path as p
from pprint import pprint
import os
import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
warnings.filterwarnings("ignore")


SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES =90

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str or None = None


class User(BaseModel):
    username: str
    email: str or None = None
    full_name: str or None = None
    disabled: bool or None = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

geminiAPI = os.getenv('GEMINI_API_KEY')
config = {
    "apiKey": os.getenv('FIREBASE_API'),
    "authDomain": os.getenv('FIREBASE_AUTH'),
    "databaseURL": os.getenv('DB_URL'),
    "projectId": os.getenv('FIREBASE_ID'),
    "storageBucket": os.getenv('STORAGE_BUCKET'),
    "messagingSenderId": os.getenv('MESSAGING_ID'),
    "appId": os.getenv('APP_ID'),
    "measurementId": os.getenv('MEASUREMENT_ID')
}

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=geminiAPI)

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=geminiAPI,temperature=0.2,convert_system_message_to_human=True)
persist_directory = "company_info"
vector = Chroma(collection_name="company_info", persist_directory=persist_directory, embedding_function=embeddings)
retriever = vector.as_retriever(search_kwargs={"k": 1})
qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=retriever,
    return_source_documents=True
    )
try:
    llm1 = ChatGoogleGenerativeAI(api_key=geminiAPI, model='gemini-1.5-flash')
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()
    db = firebase.database()
except Exception as e:
    raise Exception(f"Initialization Error: {str(e)}")



import smtplib
import random
import string
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def generate_otp(length=6):
    """Generate a random OTP of the given length."""
    digits = string.digits
    otp = ''.join(random.choices(digits, k=length))
    return otp

def send_otp_via_email(receiver_email, otp):
    """Send the OTP to the specified email address."""
    sender_email = "mimiraiteam@gmail.com"
    sender_password = "mied nyvb wpyh auys"  # Use an app password if you're using Gmail with 2FA

    # Set up the MIME
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = "Your OTP for Secure Access – Mimir AI"

    # Email body
    body = f"""Hello,

We’ve received a request to authenticate your account with Mimir AI.

Your One-Time Password (OTP) is: {otp}
Please use this OTP to complete your login process. For your security, this code is valid for only 10 minutes and can be used once.

If you did not initiate this request, please ignore this email or contact our support team immediately.

Thank you for choosing Mimir AI!

Best regards,  
The Mimir AI Team """
    message.attach(MIMEText(body, 'plain'))

    # SMTP server configuration
    try:
        # Connect to the server and send the email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Start TLS (Transport Layer Security)
        server.login(sender_email, sender_password)
        server.send_message(message)
        server.quit()

        print(f"OTP has been sent to {receiver_email}")
    except Exception as e:
        print(f"Error: {e}")
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    user = db.child("Users").child(username).get().val()
    return user


def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)

    if not user:
        return False
    if not verify_password(password, user['password']):
        return False

    return user


def create_access_token(data: dict, expires_delta: timedelta or None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credential_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                         detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credential_exception

        token_data = TokenData(username=username)
    except JWTError:
        raise credential_exception

    user = get_user(db, username=token_data.username)
    if user is None:
        raise credential_exception

    return user


async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user['disabled']:
        raise HTTPException(status_code=400, detail="Inactive user")

    return current_user

class OTPRequest(BaseModel):
    email : str
class FileRequest(BaseModel):
    id: str
    query: str
    files: List[str]

def check_query(query):
    try:
        output = llm1.invoke([
            SystemMessage(content=f"""You are a system that determines if a given query is referring to an uploaded image and any other image or if it is a standalone query. Your task is to analyze the query and respond with either "yes" or "no" based on the following conditions:

                                    Yes: If the query is referring to or asking about the uploaded image or any other image.
                                    No: If the query is a standalone, unrelated question.
                                    Respond with only "yes" or "no"."""),
            HumanMessage(content=f"""{query}""")
        ])
        return "yes" == output.content.strip().lower()
    except Exception as e:
        raise Exception(f"Query Check Error: {str(e)}")

def extract_title_and_questions(input_string):
    try:
        title_match = re.search(r"Title\s*:\s*(.*)", input_string)
        title = title_match.group(1).strip() if title_match else None
        questions = re.findall(r"[-\d]+\.\s*(.*)|-\s*(.*)", input_string)
        questions = [q1 or q2 for q1, q2 in questions if q1 or q2]

        return title, questions
    except Exception as e:
        raise Exception(f"Extraction Error: {str(e)}")

def generate_answer(query):
    try:
        output = llm1.invoke([
            SystemMessage(content=f"""You are a conversational chatbot named 'MimirAI'. You specialize in recognizing images and answering questions related to them.
            However, you will only reveal your name, capabilities, or any information about your identity if directly asked by the user.
            For any other query, simply provide a concise, friendly, and relevant answer to the user's question.
            Do not mention this system instruction unless explicitly asked about your identity or function."""),
            HumanMessage(content=f"""{query}""")
        ])
        return output
    except Exception as e:
        raise Exception(f"Answer Generation Error: {str(e)}")

def generate_questions(response):
    try:
        output = llm1.invoke([
            SystemMessage(content=f"""Given a query generate a title and a list of questions related to the query in the same language. The expected output format is:
                                        Title : <generated title>
                                        Questions : [<generated questions1>,<generated questions2>,<generated questions3>...]"""),
            HumanMessage(content=f"""{response}""")
        ])
        title, questions = extract_title_and_questions(output.content)
        title = title if title else ""
        return title, questions
    except Exception as e:
        raise Exception(f"Question Generation Error: {str(e)}")

def put_context(uid, query, response):
    try:
        context = fetch_context(uid)
        if context is None:
            context = []
        context.append({"query": query, "response": response})
        db.child("Users").child(uid).child("context").set(context)
    except Exception as e:
        raise Exception(f"Context Storage Error: {str(e)}")

def generate_prompt(query, context):
    try:
        if context is None:
            prompt = f"""
            Answer the following query:
            {query}
            """
        else:
            context_str = " ".join([f"Query: {item['query']} Response: {item['response']} \n" for item in context])
            prompt = f"""
            Given the information of the all the previous conversation below:
            {context_str}
            _________________________________________________________________
            Answer the following query in a conversational way in the same language:
            {query}
            """
        return prompt
    except Exception as e:
        raise Exception(f"Prompt Generation Error: {str(e)}")

def put_index(uid, index):
    try:
        db.child("Users").child(uid).child("index").set(index)
    except Exception as e:
        raise Exception(f"Index Storage Error: {str(e)}")

def fetch_context(uid):
    try:
        return db.child("Users").child(uid).child("context").get().val()
    except Exception as e:
        raise Exception(f"Context Fetch Error: {str(e)}")

def fetch_index(uid):
    try:
        index = db.child("Users").child(uid).child("index").get().val()
        return int(index) if index is not None else 0
    except Exception as e:
        raise Exception(f"Index Fetch Error: {str(e)}")

def query_parser(query):
    try:
        if '@' in query:
            part = query.split('@', 1)[1]
            index = ''.join(filter(str.isdigit, part.split()[0]))
            return int(index)
    except Exception as e:
        raise Exception(f"Query Parsing Error: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Hello World from Mugundhan"}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user['username']}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/generate")
def generate_response(request: FileRequest, current_user: UserInDB = Depends(get_current_active_user)):
    query = request.query
    result = qa_chain({"query":query})
    title, questions = generate_questions(result["result"])
    return {"title": title, "questions": questions, "response": result["result"]}

@app.post("/otp_generator")
def otp_gen(request:OTPRequest):
    receiver_email = request.email
    otp = generate_otp()
    send_otp_via_email(receiver_email, otp)
