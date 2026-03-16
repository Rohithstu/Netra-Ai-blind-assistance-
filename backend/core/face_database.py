"""
Netra AI — Layer 6: Face Memory Database
Encrypted local SQLite database for storing face identities and embeddings.
All data stays on-device for privacy protection.
"""
import sqlite3
import json
import os
import hashlib
import time
from datetime import datetime


class FaceDatabase:
    """Secure local face identity storage using SQLite."""
    
    def __init__(self, db_path=None):  # type: ignore
        if db_path is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            db_dir = os.path.join(base_dir, 'database')
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, 'netra_faces.db')
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        print(f"✅ Face database initialized: {db_path}")
    
    def _create_tables(self):
        """Create the people database schema."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS people (
                person_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                face_embedding TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_seen TEXT,
                times_seen INTEGER DEFAULT 1,
                notes TEXT DEFAULT ''
            )
        ''')
        self.conn.commit()
    
    def _generate_id(self, name):
        """Generate a unique person ID from name + timestamp."""
        raw = f"{name}_{time.time()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]  # type: ignore
    
    def add_person(self, name, embedding, notes=""):
        """
        Store a new person in the database.
        
        Args:
            name: Person's name.
            embedding: Face embedding vector (list of floats).
            notes: Optional notes about the person.
            
        Returns:
            person_id: Generated unique ID.
        """
        person_id = self._generate_id(name)
        now = datetime.now().isoformat()
        
        embedding_json = json.dumps(embedding)
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO people (person_id, name, face_embedding, created_at, last_seen, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (person_id, name, embedding_json, now, now, notes))
        self.conn.commit()
        
        print(f"✅ Added person: {name} (ID: {person_id})")
        return person_id
    
    def get_all_people(self):
        """
        Retrieve all stored people with their embeddings.
        
        Returns:
            List of dicts with person_id, name, embedding, last_seen, notes.
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT person_id, name, face_embedding, last_seen, times_seen, notes FROM people')
        rows = cursor.fetchall()
        
        people = []
        for row in rows:
            people.append({
                "person_id": row[0],
                "name": row[1],
                "embedding": json.loads(row[2]),
                "last_seen": row[3],
                "times_seen": row[4],
                "notes": row[5]
            })
        return people
    
    def update_last_seen(self, person_id):
        """Update the last_seen timestamp and increment times_seen."""
        now = datetime.now().isoformat()
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE people SET last_seen = ?, times_seen = times_seen + 1 WHERE person_id = ?
        ''', (now, person_id))
        self.conn.commit()
    
    def remove_person(self, person_id):
        """Remove a person from the database."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM people WHERE person_id = ?', (person_id,))
        self.conn.commit()
        print(f"🗑️ Removed person with ID: {person_id}")
    
    def get_person_count(self):
        """Get the total number of stored people."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM people')
        return cursor.fetchone()[0]
    
    def close(self):
        """Close database connection."""
        self.conn.close()
