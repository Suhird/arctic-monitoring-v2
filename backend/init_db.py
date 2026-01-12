"""
Initialize database tables and create a test user
"""
import sys
sys.path.insert(0, '/Users/suhird/Desktop/ideas/arctic-ice-monitoring-v2/backend')

from app.database import init_db, SessionLocal
from app.services import auth_service
from app.schemas.user import UserCreate

def main():
    print("Initializing database...")

    # Create tables
    init_db()
    print("✓ Database tables created")

    # Create test user
    db = SessionLocal()
    try:
        # Check if user already exists
        existing_user = db.query(auth_service.User).filter_by(email="test@arctic.com").first()

        if existing_user:
            print("✓ Test user already exists")
        else:
            user_data = UserCreate(
                email="test@arctic.com",
                password="testpass123",
                full_name="Test User"
            )
            user = auth_service.create_user(db, user_data)
            print(f"✓ Created test user: {user.email}")
    except Exception as e:
        print(f"Error creating user: {e}")
    finally:
        db.close()

    print("\n" + "="*60)
    print("Database initialization complete!")
    print("="*60)
    print("\nLogin credentials:")
    print("  Email: test@arctic.com")
    print("  Password: testpass123")
    print("\nAccess the UI at: http://localhost:3000")
    print("="*60)

if __name__ == "__main__":
    main()
