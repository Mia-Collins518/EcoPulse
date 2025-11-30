"use client"

import { SidebarTrigger } from "@/components/ui/sidebar"  // adjust path if needed

export default function Heading(){
    return(
        <header className="flex px-3 items-center justify-between bg-brand-header w-full h-17">
            <SidebarTrigger />
        
            <a href="/" className="flex items-center gap-2 hover:opacity-80 transition">
                <img src="/logo-condensed.png" alt="EcoPulse logo" width={40} height={40} />
                <p className="font-bold text-2xl text-[color:var(--primary)]">EcoPulse</p>
            </a>

        </header>
    )
}