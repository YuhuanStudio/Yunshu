import Link from "next/link";

export default function NotFound() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-4 p-8 text-center">
      <h1 className="text-2xl font-semibold">404</h1>
      <Link href="/" className="text-sm underline underline-offset-4">
        Yunshu Docs
      </Link>
    </main>
  );
}
