import Image from "next/image";
import Advisor from "./(adivsor)/Advisor";

export default function Home() {
    return (
        <main className="items-center justify-between">
            <div>
                <Advisor />
            </div>
        </main>
    );
}
