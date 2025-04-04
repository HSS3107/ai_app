# Generated by Django 5.1.1 on 2024-12-25 13:54

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("app", "0006_videometadata_remove_chatsession_content_map_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="chatsession",
            name="resource",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="chat_sessions",
                to="app.resource",
            ),
        ),
    ]
