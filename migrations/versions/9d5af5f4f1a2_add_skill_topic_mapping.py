"""add skill topic mapping

Revision ID: 9d5af5f4f1a2
Revises: 641637972ffe
Create Date: 2026-03-11 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "9d5af5f4f1a2"
down_revision = "641637972ffe"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "skills",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("skill_id", sa.String(length=120), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("skill_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("prerequisite_skill_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["prerequisite_skill_id"], ["skills.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("skills", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_skills_skill_id"), ["skill_id"], unique=True)
        batch_op.create_index(
            batch_op.f("ix_skills_prerequisite_skill_id"),
            ["prerequisite_skill_id"],
            unique=False,
        )

    op.create_table(
        "skill_topic_mapping",
        sa.Column("skill_id", sa.Integer(), nullable=False),
        sa.Column("topic_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["skill_id"], ["skills.id"]),
        sa.ForeignKeyConstraint(["topic_id"], ["topics.id"]),
        sa.PrimaryKeyConstraint("skill_id", "topic_id"),
    )


def downgrade():
    op.drop_table("skill_topic_mapping")

    with op.batch_alter_table("skills", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_skills_prerequisite_skill_id"))
        batch_op.drop_index(batch_op.f("ix_skills_skill_id"))

    op.drop_table("skills")
