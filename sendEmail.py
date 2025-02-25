import smtplib # for sending email
from email.mime.text import MIMEText # for sending email
from email.mime.multipart import MIMEMultipart # for sending email
import os
from email.mime.base import MIMEBase
from email import encoders
from email.mime.image import MIMEImage

def sendEmail(loops, trainingLoss, validationLoss, trainingComplete, fileName, accuracy = 0.0):
    # Email details
    sender_email = "random free gmail :)"
    receiver_email = "nope"
    password = "nice try"  # I use an app password through Google Account to allow Python access

    # Set up the server
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()  # Secure the connection
    server.login(sender_email, password)

    # Create email content
    
    subject = str(loops) + " loops completed"
    if trainingComplete:
        subject += " -- Training Completed"
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    if accuracy > 0.0:
        message.attach(MIMEText(f"accuracy: {accuracy} -- {round(trainingLoss, 2)} trainingLoss, {round(validationLoss, 2)} validation loss", "plain"))
    else:
        message.attach(MIMEText(f"{round(trainingLoss, 2)} trainingLoss, {round(validationLoss, 2)} validation loss", "plain"))


    # Only attach the file if training is complete
    if trainingComplete and os.path.exists(fileName):
        if fileName.lower().endswith(".png"):
            # Attach as PNG image using MIMEImage
            with open(fileName, "rb") as attachment:
                img = MIMEImage(attachment.read(), _subtype="png")
                img.add_header("Content-Disposition", f"attachment; filename={os.path.basename(fileName)}")
                message.attach(img)
        else:
            # Attach as a generic file type
            with open(fileName, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(fileName)}")
            message.attach(part)
    server.sendmail(sender_email, receiver_email, message.as_string())
    # server.sendmail(sender_email, "group mate email", message.as_string())
    server.quit()
