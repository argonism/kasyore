"""create esa_docs

Revision ID: 0e8618696745
Revises: 
Create Date: 2024-03-16 01:20:01.166906

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0e8618696745'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "esa_docs",
        sa.Column("number", sa.String(length=10), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("full_name", sa.Text(), nullable=True),
        sa.Column("wip", sa.Boolean(), nullable=True),
        sa.Column("body_md", sa.Text(), nullable=True),
        sa.Column("body_html", sa.Text(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(length=20), nullable=True),
        sa.PrimaryKeyConstraint("number"),
    )


def downgrade() -> None:
    op.drop_table('esa_docs')
