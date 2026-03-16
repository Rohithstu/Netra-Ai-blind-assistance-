"""
Netra AI — Layer 6: Face Memory Manager CLI
Command-line tool for managing the face identity database.
Supports adding, listing, removing, and interactively teaching new faces.
"""
import cv2  # type: ignore
import argparse
import os
import sys

from face_database import FaceDatabase

# Import face engine for embedding extraction
try:
    from face_engine import FaceEngine
except ImportError:
    FaceEngine = None


def list_people(db):
    """List all stored people in the database."""
    people = db.get_all_people()
    
    if len(people) == 0:
        print("📭 No people stored yet. Use --add to remember someone.")
        return
    
    print(f"\n👥 Known People ({len(people)}):")
    print("-" * 60)
    for p in people:
        print(f"  📌 {p['name']}")
        print(f"     ID: {p['person_id']}")
        print(f"     Last seen: {p['last_seen']}")
        print(f"     Times seen: {p['times_seen']}")
        if p['notes']:
            print(f"     Notes: {p['notes']}")
        print()


def add_person_interactive(db):
    """Interactively capture a face from webcam and store it."""
    if FaceEngine is None:
        print("❌ Face engine not available. Ensure face_engine.py is in the same directory.")
        return
    
    name = input("📝 Enter the person's name: ").strip()
    if not name:
        print("❌ Name cannot be empty.")
        return
    
    notes = input("📝 Any notes? (press Enter to skip): ").strip()
    
    print(f"\n🎥 Opening camera... Please position {name}'s face clearly in the frame.")
    print("Press SPACE to capture or 'q' to cancel.\n")
    
    cap = cv2.VideoCapture(0)
    captured = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display preview
        display = frame.copy()
        cv2.putText(display, f"Capturing: {name} — Press SPACE to capture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Netra AI — Face Capture", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            # Attempt to capture and store
            engine = FaceEngine(use_depth=False)
            success = engine.remember_person(frame, name, notes)
            if success:
                print(f"\n🎉 {name} has been remembered! Netra will recognize them in the future.")
                captured = True
            break
        elif key == ord('q'):
            print("❌ Capture cancelled.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if not captured:
        print("⚠️ No face was stored.")


def remove_person_interactive(db):
    """Remove a person from the database by selecting from the list."""
    people = db.get_all_people()
    
    if len(people) == 0:
        print("📭 No people stored.")
        return
    
    print(f"\n👥 Stored People:")
    for i, p in enumerate(people):
        print(f"  [{i+1}] {p['name']} (ID: {p['person_id'][:8]}...)")
    
    try:
        choice = int(input("\nEnter number to remove (0 to cancel): "))
        if choice == 0:
            print("Cancelled.")
            return
        if 1 <= choice <= len(people):
            person = people[choice - 1]
            confirm = input(f"Are you sure you want to remove {person['name']}? (y/n): ").strip().lower()
            if confirm == 'y':
                db.remove_person(person['person_id'])
                print(f"✅ {person['name']} has been removed.")
            else:
                print("Cancelled.")
        else:
            print("❌ Invalid selection.")
    except ValueError:
        print("❌ Invalid input.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Netra AI — Face Memory Manager")
    parser.add_argument('--list', action='store_true', help="List all stored people")
    parser.add_argument('--add', action='store_true', help="Add a new person via webcam capture")
    parser.add_argument('--remove', action='store_true', help="Remove a person from the database")
    parser.add_argument('--count', action='store_true', help="Show number of stored people")
    args = parser.parse_args()
    
    db = FaceDatabase()
    
    if args.list:
        list_people(db)
    elif args.add:
        add_person_interactive(db)
    elif args.remove:
        remove_person_interactive(db)
    elif args.count:
        count = db.get_person_count()
        print(f"👥 Total stored people: {count}")
    else:
        # Default: show menu
        print("Netra AI — Face Memory Manager")
        print("Usage:")
        print("  --list     List all stored people")
        print("  --add      Add a new person via webcam")
        print("  --remove   Remove a stored person")
        print("  --count    Show total count")
    
    db.close()
