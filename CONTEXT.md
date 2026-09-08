# Recon (AI Facial OSINT Reconnaissance Engine)

ระบบสแกน ค้นหา และวิเคราะห์ใบหน้าบุคคลเป้าหมายบนเว็บไซต์และคลังภาพระดับองค์กรด้วย AI Vision (RetinaFace + ArcFace) และ Qdrant Vector Database

## Language

**Target Face**:
ภาพหรือเวกเตอร์ของบุคคลที่เป็นเป้าหมายในการค้นหา ซึ่งผู้ใช้อัปโหลดเข้ามาในระบบ
_Avoid_: Reference photo, query picture, input face

**Candidate Face**:
ใบหน้าที่ตรวจพบบนเว็บเพจหรือในคลังภาพ และนำมาคำนวณเปรียบเทียบกับ Target Face
_Avoid_: Match candidate, detected person, suspect

**Face Embedding**:
เวกเตอร์ตัวเลขทศนิยมขนาด 512 มิติ (Normalized float32) ที่สกัดจากโมเดล ArcFace w600k_r50 เพื่อแทนอัตลักษณ์ทางกายภาพของใบหน้า
_Avoid_: Feature vector, facial hash, signature

**Cosine Similarity**:
ค่าผลคูณ Dot Product ระหว่าง Normalized Embeddings สองตัว (ช่วงค่า -1.0 ถึง 1.0) แทนระดับความเหมือนกันของบุคคล
_Avoid_: Distance, match percent, confidence score

**Score Threshold**:
เกณฑ์คะแนน Cosine Similarity ขั้นต่ำ (แนะนำที่ 0.50) ที่กำหนดให้ระบบยอมรับ Candidate Face ว่าตรงกับ Target Face
_Avoid_: Cutoff, tolerance, sensitivity level

**Direct Max Similarity**:
วิธีการตัดสินผลการจับคู่โดยใช้ค่า Similarity สูงสุดระหว่าง Candidate กับภาพมุมมองต่างๆ ของบุคคลเป้าหมายโดยตรง
_Avoid_: Centroid averaging, blended similarity, mean vector

**Multi-Scale Tiling**:
กระบวนการตัดแบ่งภาพความละเอียดสูงออกเป็นโซนย่อยแบบเหลื่อมซ้อน (Quadrants) เพื่อตรวจจับใบหน้าขนาดเล็กที่ยืนระยะไกล
_Avoid_: Window slicing, sub-grid scan, sliding crop

**Full-Site Crawler**:
บอทสำรวจเว็บแบบ Asynchronous ที่ทำหน้าที่ขุดทุกลิงก์ภายในโดเมนเดียวกัน และแกะลิงก์รูปภาพต้นฉบับคุณภาพสูง (Unwrapped High-Res)
_Avoid_: Scraper, web spider, image downloader

**Offline Dataset**:
ชุดข้อมูลรูปภาพและ Metadata ที่แคชไว้บน SSD ในเครื่องสำหรับการทดสอบและสแกนด้วย GPU แบบความเร็วสูงโดยไม่ต้องพึ่งพาระบบเครือข่าย
_Avoid_: Local cache, offline folder, local storage
