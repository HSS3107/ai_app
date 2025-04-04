# Generated by Django 5.1.1 on 2024-12-24 15:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("app", "0004_chatsession_content_map_chatsession_video_id_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="WaitList",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.TextField()),
                ("email", models.EmailField(max_length=254, unique=True)),
                ("phone_no", models.CharField(max_length=15, unique=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "db_table": "app_waitlist",
            },
        ),
    ]
