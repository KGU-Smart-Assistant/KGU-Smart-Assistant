import Link from "next/link";

export default function BottomNav() {
  return (
    <nav
      style={{
        position: "fixed",
        bottom: 0,
        left: "50%",
        transform: "translateX(-50%)",
        width: "100%",
        maxWidth: "430px",
        boxSizing: "border-box",
        display: "flex",
        justifyContent: "space-around",
        padding: "16px",
        backgroundColor: "#003876", // 여기 변경
        borderTop: "1px solid #333",
        zIndex: 100,
  }}
>
      <Link href="/" style={{ color: "white", textDecoration: "none" }}>
        홈
      </Link>
      <Link href="/map" style={{ color: "white", textDecoration: "none" }}>
        지도
      </Link>
      <Link href="/phone" style={{ color: "white", textDecoration: "none" }}>
        전화번호
      </Link>
    </nav>
  );
}
