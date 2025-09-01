from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

class Book(BaseModel):
    title: str
    author: str
    year: int

books: Dict[int, Book] = {}
book_id_counter = 1

@app.post("/books")
def create_book(book: Book):
    global book_id_counter
    books[book_id_counter] = book
    book_id_counter += 1
    return {"id": book_id_counter - 1, "book": book}

@app.get("/books")
def get_books():
    return books

@app.get("/books/{book_id}")
def get_book(book_id: int):
    if book_id not in books:
        raise HTTPException(status_code=404, detail="Book not found")
    return books[book_id]


@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI app 🚀"}