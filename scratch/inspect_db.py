import os
import asyncio
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("POSTGRES_DB_URL")

async def inspect():
    from sqlalchemy.ext.asyncio import create_async_engine
    if DB_URI.startswith("postgresql://"):
        async_uri = DB_URI.replace("postgresql://", "postgresql+psycopg://", 1)
    else:
        async_uri = DB_URI
        
    engine = create_async_engine(async_uri)
    async with engine.connect() as conn:
        # Get list of tables
        res = await conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public';"))
        tables = [r[0] for r in res.fetchall()]
        print("Tables in database:", tables)
        
        for table in tables:
            print(f"\nColumns in {table}:")
            cols_res = await conn.execute(text(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = :t"), {"t": table})
            for col in cols_res.fetchall():
                print(f"  {col[0]}: {col[1]}")
                
            # Sample data
            try:
                sample_res = await conn.execute(text(f"SELECT * FROM {table} LIMIT 1"))
                row = sample_res.fetchone()
                if row:
                    print(f"  Sample row from {table}:", dict(row._mapping))
            except Exception as e:
                print(f"  Could not get sample: {e}")

if __name__ == "__main__":
    asyncio.run(inspect())
